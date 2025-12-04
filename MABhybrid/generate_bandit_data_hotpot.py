import os
import sys
import json
import math
import numpy as np
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import pytrec_eval
from collections import defaultdict


# ------Configuration------
DATASET = "hotpotqa"
OUTPUT_FILE = "../MABhybrid/data/bandit_data_train_hotpot.jsonl"

# Retrieval & feature config
MAX_DOCS = 200000  # subsample docs from HotpotQA corpus to keep things tractable
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 100       # depth per retriever
EVAL_K = 10       # nDCG@10
SAMPLE_SIZE = 7000  # number of queries to sample
ARMS = [0.0, 0.25, 0.5, 0.75, 1.0]  # alpha values: 0.0=dense-only, 1.0=sparse-only


def simple_tokenize(text: str):
    return text.lower().split()


def normalize_scores_list(hits):
    """
    hits: list of (docid, score)
    returns dict {docid: norm_score} with min-max normalization
    """
    if not hits:
        return {}
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {d: 1.0 for d, _ in hits}
    return {d: (s - lo) / (hi - lo) for d, s in hits}


def calculate_ndcg(run_dict, qrels_dict, k=EVAL_K):
    """
    run_dict: {docid: score}
    qrels_dict: {docid: relevance}
    """
    if not qrels_dict:
        return 0.0
    evaluator = pytrec_eval.RelevanceEvaluator({'q': qrels_dict}, {'ndcg_cut'})
    results = evaluator.evaluate({'q': run_dict})
    return results['q'][f'ndcg_cut_{k}']



def main():
    # ---------------- BEIR / HotpotQA data ----------------
    print(f"Downloading/loading BEIR dataset: {DATASET}...")
    out_dir = "./datasets"
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"

    data_path = util.download_and_unzip(url, out_dir)
    print("Data path:", data_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded corpus with {len(corpus)} docs, {len(queries)} queries.")

    # ---------------- Subsample corpus ----------------
    all_doc_ids = list(corpus.keys())
    np.random.seed(42)
    num_docs = min(MAX_DOCS, len(all_doc_ids))
    sub_doc_ids = np.random.choice(all_doc_ids, size=num_docs, replace=False).tolist()
    print(f"Using {len(sub_doc_ids)} docs out of {len(all_doc_ids)} total.")

    doc_ids = sub_doc_ids
    doc_texts = []

    for did in doc_ids:
        title = corpus[did].get("title") or ""
        text = corpus[did].get("text") or ""
        full_text = (title + " " + text).strip()
        if len(full_text) > 1000:
            full_text = full_text[:1000]
        if not full_text:
            full_text = "[EMPTY DOC]"
        doc_texts.append(full_text)

    # ---------------- BM25 index ----------------
    print("Tokenizing corpus for BM25...")
    tokenized_corpus = [simple_tokenize(t) for t in doc_texts]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # ---------------- Dense index ----------------
    print("Loading dense model:", DENSE_MODEL_NAME)
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    print("Encoding subsampled corpus for dense retrieval...")
    doc_embeddings = dense_model.encode(
        doc_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = doc_embeddings.shape[1]
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    print("Subsampled corpus ready for BM25 + dense search.")

    # ---------------- Build IDF statistics (FeatureExtractor-equivalent) ----------------
    print("Building term document frequencies for IDF...")
    df_counts = defaultdict(int)
    for tokens in tokenized_corpus:
        unique_terms = set(tokens)
        for t in unique_terms:
            df_counts[t] += 1
    N = len(doc_ids)
    print("IDF table ready. N =", N, "unique terms:", len(df_counts))

    def idf(term):
        df = df_counts.get(term, 0)
        return float(np.log(N / (df + 1)))

    def extract_features(query_text):
        """
        5-d feature vector (aligned with original MABhybrid FeatureExtractor):

        1. Length        : number of tokens
        2. Max IDF       : rarity of the rarest term
        3. Avg IDF       : average rarity
        4. QuestionFlag  : 1.0 if starts with Wh-word, else 0.0
        5. Bias          : constant 1.0
        """
        tokens = simple_tokenize(query_text)
        length = len(tokens)
        if length == 0:
            return [0.0, 0.0, 0.0, 0.0, 1.0]

        idfs = [idf(t) for t in tokens]
        max_idf = max(idfs)
        avg_idf = float(np.mean(idfs))

        question_starters = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
        is_question = 1.0 if tokens[0] in question_starters else 0.0

        return [
            float(length),
            float(max_idf),
            float(avg_idf),
            float(is_question),
            1.0
        ]

    # ---------------- Search wrappers ----------------
    def sparse_search(query_text, k=TOP_K):
        toks = simple_tokenize(query_text)
        scores = bm25.get_scores(toks)
        top_idx = np.argsort(scores)[::-1][:k]
        hits = [(doc_ids[i], float(scores[i])) for i in top_idx]
        return hits

    def dense_search(query_text, k=TOP_K):
        q_emb = dense_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = index.search(q_emb, k)
        hits = [(doc_ids[idx], float(score)) for score, idx in zip(scores[0], idxs[0])]
        return hits

    # ---------------- Valid queries & sampling ----------------
    valid_qids = [qid for qid in queries.keys() if qid in qrels]
    print("Valid queries (with qrels):", len(valid_qids))

    num_samples = min(SAMPLE_SIZE, len(valid_qids))
    np.random.seed(42)
    sampled_qids = list(np.random.choice(valid_qids, num_samples, replace=False))
    print(f"Sampling {len(sampled_qids)} queries for bandit data.")

    # ---------------- Main bandit data generation loop ----------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f_out:
        for qid in tqdm(sampled_qids, desc="Generating bandit data"):
            qtext = queries[qid]

            try:
                sparse_hits = sparse_search(qtext, k=TOP_K)
                dense_hits = dense_search(qtext,  k=TOP_K)
            except Exception:
                # Skip weird queries if something goes wrong
                continue

            sparse_dict = normalize_scores_list(sparse_hits)
            dense_dict = normalize_scores_list(dense_hits)
            all_docs = set(sparse_dict.keys()) | set(dense_dict.keys())

            qrels_for_query = qrels.get(qid, {})
            rewards = []

            # Fusion and reward for each arm
            for alpha in ARMS:
                fused_scores = {}
                for docid in all_docs:
                    s = sparse_dict.get(docid, 0.0)
                    d = dense_dict.get(docid, 0.0)
                    fused_scores[docid] = alpha * s + (1.0 - alpha) * d

                # Limit to top 100 docs for evaluation speed
                top_100 = dict(
                    sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:100]
                )
                r = calculate_ndcg(top_100, qrels_for_query, k=EVAL_K)
                rewards.append(float(r))

            if not rewards:
                continue

            features = extract_features(qtext)

            record = {
                "query_id": qid,
                "text": qtext,
                "features": features,
                "rewards": rewards,
                "optimal_arm": int(np.argmax(rewards))
            }
            f_out.write(json.dumps(record) + "\n")

    print(f"Done! Generated HotpotQA bandit data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()