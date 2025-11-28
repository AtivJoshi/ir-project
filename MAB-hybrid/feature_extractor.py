import numpy as np

class FeatureExtractor:
    """
    Converts raw text queries into dense numerical vectors using Pyserini 
    for corpus statistics.
    """
    def __init__(self, pyserini_index_reader):
        """
        Args:
            pyserini_index_reader: An instance of pyserini.index.IndexReader
                                   pointing to the MS MARCO index.
        """
        self.reader = pyserini_index_reader
        # Total number of documents in the collection (N) for IDF calculation
        self.N = self.reader.stats()['documents']

    def get_idf(self, term):
        """
        Calculates Inverse Document Frequency (IDF) for a single term.
        Formula: log( N / (df + 1) )
        """
        # analyze=True ensures the term is stemmed/tokenized consistent with the index
        df, _ = self.reader.get_term_counts(term, analyzer=None)
        
        # Avoid division by zero or log(0) issues
        if df > 0:
            return np.log(self.N / (df + 1))
        return 0.0

    def extract(self, query_text):
        """
        Extracts the 5-dimensional feature vector for a query.
        
        Feature Definition:
        1. Length: Number of tokens
        2. Max IDF: Rarity of the rarest word (keyword specificity)
        3. Avg IDF: Average rarity (information density)
        4. Question Flag: 1.0 if starts with Wh-word, else 0.0
        5. Bias: Constant 1.0 (Intercept)
        """
        # Simple whitespace tokenization (can be improved with a proper tokenizer)
        tokens = query_text.lower().split()
        length = len(tokens)
        
        if length == 0:
            return np.array([0, 0, 0, 0, 1.0])
        
        # Compute IDFs
        idfs = [self.get_idf(t) for t in tokens]
        
        max_idf = max(idfs) if idfs else 0.0
        avg_idf = np.mean(idfs) if idfs else 0.0
        
        # Heuristic: Check for Wh-words to detect natural language questions
        question_starters = {'who', 'what', 'where', 'when', 'why', 'how', 'which'}
        is_question = 1.0 if tokens[0] in question_starters else 0.0
        
        # Construct and return the vector
        # [Length, MaxIDF, AvgIDF, QuestionFlag, Bias]
        return np.array([float(length), max_idf, avg_idf, is_question, 1.0])