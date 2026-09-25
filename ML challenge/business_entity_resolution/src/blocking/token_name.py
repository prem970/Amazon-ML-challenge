"""
Token-based name blocking keys (B3).
Indexes rare, distinctive, high-information name tokens to recover word-order and partial name variations.
"""

from typing import Dict, List, Set
from collections import defaultdict

GENERIC_NAME_STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'all', 'new', 'one', 'two',
    'center', 'centre', 'group', 'services', 'service', 'solutions', 'solution',
    'enterprises', 'enterprise', 'trading', 'consulting', 'investments', 'investment',
    'associates', 'international', 'global', 'holdings', 'holding', 'management',
    'industries', 'industry', 'corporation', 'company', 'limited', 'private',
    'systems', 'system', 'tech', 'technologies', 'technology', 'care', 'health',
    'store', 'shop', 'mart', 'market', 'hotel', 'restaurant', 'cafe', 'agency'
}

class TokenNameBlocker:
    def __init__(self, min_token_len: int = 4, max_bucket_size: int = 150):
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.token_index = defaultdict(list)

    def extract_distinctive_tokens(self, tokens: List[str]) -> List[str]:
        distinctive = []
        for t in tokens:
            t_clean = t.strip().lower()
            if len(t_clean) >= self.min_token_len and t_clean not in GENERIC_NAME_STOPWORDS:
                distinctive.append(t_clean)
        # Sort by length descending (longer tokens usually carry higher information)
        distinctive.sort(key=len, reverse=True)
        return distinctive[:2]

    def add_record(self, entity_id: str, tokens: List[str]):
        distinctive = self.extract_distinctive_tokens(tokens)
        for t in distinctive:
            if len(self.token_index[t]) < self.max_bucket_size:
                self.token_index[t].append(entity_id)

    def get_candidates(self, tokens: List[str]) -> Set[str]:
        candidates = set()
        distinctive = self.extract_distinctive_tokens(tokens)
        for t in distinctive:
            if t in self.token_index:
                candidates.update(self.token_index[t])
        return candidates
