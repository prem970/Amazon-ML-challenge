"""
Precision safety layer and singleton / abstention logic.
Enforces precision guards, contradictory evidence rejection, and singleton abstention.
"""

from typing import List, Dict, Set, Tuple
try:
    from ..config import SAFETY_CONFIG
except (ImportError, ValueError):
    from config import SAFETY_CONFIG

class PrecisionSafetyLayer:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def filter_entity_matches(
        self,
        candidate_scores: List[Tuple[str, float, Dict[str, any]]]  # (cand_id, prob, metadata)
    ) -> List[str]:
        """
        Apply precision-first decision logic for a single Source 1 entity:
        1. If top candidate < threshold -> empty list (singleton)
        2. Reject candidates that contradict country unless evidence is overwhelmingly high (>0.95)
        3. Allow zero, one, or multiple matches satisfying threshold
        """
        if not candidate_scores:
            return []

        # Sort descending by probability
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_prob = candidate_scores[0][1]

        # Singleton rule: if even the top candidate does not meet threshold, abstain!
        if top_prob < self.threshold:
            return []

        accepted = []
        for cand_id, prob, meta in candidate_scores:
            if prob < self.threshold:
                continue

            # Contradictory country guard
            if SAFETY_CONFIG.get("contradictory_country_penalty", True):
                c1 = meta.get("s1_country", "").strip().lower()
                c2 = meta.get("cand_country", "").strip().lower()
                if c1 and c2 and c1 != c2:
                    # Require exceptionally high confidence to merge across distinct countries
                    if prob < 0.95:
                        continue

            accepted.append(cand_id)

        return accepted
