"""
TF-IDF and character n-gram lexical retrieval blocker (B7, B8, B9).
Retrieves top-K candidates using sparse TF-IDF cosine similarity or character n-gram overlaps.
Optimized for low-memory execution.
"""

from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TFIDFRetriever:
    def __init__(self, max_features: int = 15000, top_k: int = 5):
        self.max_features = max_features
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.is_fitted = False
        self.target_matrix = None
        self.target_ids: List[str] = []

    def fit_target(self, texts: List[str], target_ids: List[str]):
        """Fit vectorizer on target records (S2/S3) and compute sparse matrix."""
        if not texts:
            return
        self.target_ids = target_ids
        self.target_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def query_top_k(self, query_texts: List[str]) -> List[List[str]]:
        """Compute cosine similarity against target matrix and return top_k target IDs for each query."""
        if not self.is_fitted or self.target_matrix is None or not query_texts:
            return [[] for _ in query_texts]

        query_vecs = self.vectorizer.transform(query_texts)
        # Dot product with normalized TF-IDF yields cosine similarity
        sim_scores = query_vecs.dot(self.target_matrix.T)

        results = []
        for i in range(sim_scores.shape[0]):
            row = sim_scores.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue
            # Get top indices
            col_indices = row.indices
            data = row.data
            if len(data) > self.top_k:
                top_part = np.argpartition(data, -self.top_k)[-self.top_k:]
                best_indices = col_indices[top_part]
            else:
                best_indices = col_indices
            results.append([self.target_ids[idx] for idx in best_indices])

        return results
