"""
Hard negative mining loop.
Identifies high-probability false positives from candidate pools to sharpen classifier boundaries.
"""

from typing import List, Tuple, Set
import numpy as np

def mine_hard_negatives(
    candidate_pairs: List[Tuple[str, str]],
    model_probs: np.ndarray,
    ground_truth: dict,
    min_prob_threshold: float = 0.35,
    max_mined: int = 10000,
) -> List[Tuple[str, str]]:
    """Find candidate non-matches that received high model prediction probabilities."""
    hard_negs = []
    for idx, (s1_id, cand_id) in enumerate(candidate_pairs):
        true_set = ground_truth.get(s1_id, set())
        # If it is NOT a true match, but the model scored it high -> Hard Negative!
        if cand_id not in true_set:
            if model_probs[idx] >= min_prob_threshold:
                hard_negs.append((s1_id, cand_id, model_probs[idx]))

    # Sort descending by probability (most deceptive first)
    hard_negs.sort(key=lambda x: x[2], reverse=True)
    return [(item[0], item[1]) for item in hard_negs[:max_mined]]
