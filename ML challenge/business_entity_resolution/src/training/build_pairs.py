"""
Training pair construction module.
Generates balanced positive and negative (random + structural hard negative) pairs.
"""

from typing import Dict, List, Set, Tuple, Any
import random
import numpy as np
try:
    from ..config import RANDOM_SEED
except (ImportError, ValueError):
    from config import RANDOM_SEED

def build_training_pairs(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, List[str]],
    s1_ids_subset: Set[str],
    neg_to_pos_ratio: float = 2.5,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Construct labeled (s1_id, cand_id) pairs and binary label vector.
    Positives: true matching pairs.
    Negatives: candidate non-matches (hard negatives from blocking + random candidates).
    """
    rng = random.Random(seed)
    pairs = []
    labels = []

    for s1_id in s1_ids_subset:
        true_matches = ground_truth.get(s1_id, set())
        cand_list = candidates.get(s1_id, [])
        cand_set = set(cand_list)

        # Positives
        for mid in true_matches:
            pairs.append((s1_id, mid))
            labels.append(1)

        # Negatives: non-matches found in candidate set
        neg_candidates = list(cand_set - true_matches)
        if neg_candidates:
            # Sample negatives proportional to positives
            num_negs = max(1, int(len(true_matches) * neg_to_pos_ratio)) if true_matches else 2
            sampled_negs = rng.sample(neg_candidates, min(len(neg_candidates), num_negs))
            for nid in sampled_negs:
                pairs.append((s1_id, nid))
                labels.append(0)

    return pairs, np.array(labels, dtype=np.int32)
