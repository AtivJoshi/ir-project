import json
import numpy as np
import sys
import os
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
import pytrec_eval
from pyserini.encode import TctColBertQueryEncoder

# Add parent directory to path so we can import MAB modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from MABhybrid.feature_extractor import FeatureExtractor
except ImportError:
    print("Error: Could not import MAB.feature_extractor. Ensure MAB/feature_extractor.py exists.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_FILE = "../MABhybrid/data/bandit_data_train.jsonl"
QUERIES_FILE = "../MABhybrid/data/msmarco-train-queries.tsv"
QRELS_FILE = "../MABhybrid/data/qrels.train.tsv"
SAMPLE_SIZE = 50000  # Number of queries to process
TOP_K = 1000         # Depth of initial retrieval
ARMS = [0.0, 0.25, 0.5, 0.75, 1.0] # Alpha values: 0.0=Dense, 1.0=Sparse

def load_queries(path, limit=None):
    """Loads queries from a TSV file (qid \t text)."""
    queries = []
    print(f"Loading queries from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            qid, text = line.strip().split('\t')
            queries.append((qid, text))
    return queries

def load_qrels(path):
    """Loads qrels into a dictionary: {qid: {docid: relevance}}."""
    qrels = {}
    print(f"Loading qrels from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            if qid not in qrels: qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    return qrels

def normalize_scores(hits):
    """Min-Max normalization of scores to [0, 1]."""
    if not hits: return {}
    scores = [h.score for h in hits]
    min_s = min(scores)
    max_s = max(scores)
    
    # Avoid division by zero if all scores are identical
    if max_s == min_s: 
        return {h.docid: 1.0 for h in hits}
        
    # Return dictionary {docid: normalized_score}
    norm_scores = {}
    for h in hits:
        norm_scores[h.docid] = (h.score - min_s) / (max_s - min_s)
    return norm_scores

def calculate_ndcg(run_dict, qrels_dict, k=10):
    """
    Calculates NDCG@k using pytrec_eval.
    run_dict: {docid: score}
    qrels_dict: {docid: relevance}
    """
    if not qrels_dict: return 0.0
    
    # pytrec_eval expects {qid: {docid: score}} structure
    evaluator = pytrec_eval.RelevanceEvaluator({'q': qrels_dict}, {'ndcg_cut'})
    results = evaluator.evaluate({'q': run_dict})
    return results['q'][f'ndcg_cut_{k}']

def main():
    # 1. Setup Pyserini Searchers
    print("Initializing Sparse Searcher (BM25)...")
    # Automatically downloads the pre-built index
    sparse_searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
    
    print("Initializing Dense Searcher (TCT-ColBERT)...")
    # Automatically downloads encoder and index
    encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
    dense_searcher = FaissSearcher.from_prebuilt_index(
        'msmarco-passage-tct_colbert-hnsw', 
        encoder
    )
    print("Dense Searcher")

    # 2. Setup Feature Extractor
    # Reuse the sparse searcher's index reader for IDF stats
    feature_extractor = FeatureExtractor(sparse_searcher)

    # 3. Load Data
    all_queries = load_queries(QUERIES_FILE, limit=None) # Load all first, then sample
    all_qrels = load_qrels(QRELS_FILE)

    # Randomly sample queries that actually have qrels
    valid_queries = [q for q in all_queries if q[0] in all_qrels]
    print(f"Found {len(valid_queries)} queries with judgments.")
    
    # Shuffle and take 50k
    np.random.seed(42)
    indices = np.random.choice(len(valid_queries), min(SAMPLE_SIZE, len(valid_queries)), replace=False)
    sampled_queries = [valid_queries[i] for i in indices]

    print(f"Starting processing for {len(sampled_queries)} queries...")
    
    with open(OUTPUT_FILE, 'w') as f_out:
        for qid, text in tqdm(sampled_queries):
            # A. Retrieve Sparse & Dense
            try:
                sparse_hits = sparse_searcher.search(text, k=TOP_K)
                dense_hits = dense_searcher.search(text, k=TOP_K)
            except Exception as e:
                # Handle empty queries or encoding errors
                continue

            # B. Normalize Scores
            # Map: {docid: norm_score}
            sparse_dict = normalize_scores(sparse_hits)
            dense_dict = normalize_scores(dense_hits)
            
            # Union of all retrieved docs
            all_docs = set(sparse_dict.keys()) | set(dense_dict.keys())
            
            query_rewards = []
            
            # C. Fusion Loop (Calculate reward for each Arm)
            for alpha in ARMS:
                fused_scores = {}
                for docid in all_docs:
                    s_score = sparse_dict.get(docid, 0.0)
                    d_score = dense_dict.get(docid, 0.0)
                    # Fusion Formula
                    score = alpha * s_score + (1.0 - alpha) * d_score
                    fused_scores[docid] = score
                
                # Sort top K for evaluation
                # Note: pytrec_eval handles sorting, but we can limit size for speed
                top_100 = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:100])
                
                # Calculate Reward (NDCG@10)
                reward = calculate_ndcg(top_100, all_qrels[qid], k=10)
                query_rewards.append(reward)

            # D. Feature Extraction
            features = feature_extractor.extract(text).tolist()

            # Save
            record = {
                "query_id": qid,
                "text": text,
                "features": features,
                "rewards": query_rewards,
                "optimal_arm": int(np.argmax(query_rewards))
            }
            f_out.write(json.dumps(record) + '\n')

    print(f"Done! Generated data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()