"""
Exact and prefix name blocking keys (B1 and B2).
B1: Exact normalized-name key (very high precision).
B2: Compact-name prefix (handles typos, punctuation, suffix variations).
"""

from typing import Dict, List, Set, Iterable
from collections import defaultdict

class ExactNameBlocker:
    def __init__(self, prefix_len: int = 8, max_bucket_size: int = 200):
        self.prefix_len = prefix_len
        self.max_bucket_size = max_bucket_size
        self.exact_index = defaultdict(list)
        self.prefix_index = defaultdict(list)

    def add_record(self, entity_id: str, legal_name: str, compact_name: str):
        if legal_name:
            if len(self.exact_index[legal_name]) < self.max_bucket_size:
                self.exact_index[legal_name].append(entity_id)

        if compact_name and len(compact_name) >= self.prefix_len:
            pfx = compact_name[:self.prefix_len]
            if len(self.prefix_index[pfx]) < self.max_bucket_size:
                self.prefix_index[pfx].append(entity_id)

    def get_candidates(self, legal_name: str, compact_name: str) -> Set[str]:
        candidates = set()
        if legal_name in self.exact_index:
            candidates.update(self.exact_index[legal_name])
        if compact_name and len(compact_name) >= self.prefix_len:
            pfx = compact_name[:self.prefix_len]
            if pfx in self.prefix_index:
                candidates.update(self.prefix_index[pfx])
        return candidates
