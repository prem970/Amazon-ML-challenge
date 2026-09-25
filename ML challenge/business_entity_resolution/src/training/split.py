"""
Entity-aware data splitting.
Splits by complete Source 1 entity IDs so candidate pairs from the same business never leak into both train and validation.
"""

from typing import List, Set, Tuple
import numpy as np
try:
    from ..config import RANDOM_SEED
except (ImportError, ValueError):
    from config import RANDOM_SEED

def split_source1_entities(
    s1_ids: List[str],
    train_ratio: float = 0.80,
    seed: int = RANDOM_SEED
) -> Tuple[Set[str], Set[str]]:
    """Split unique S1 IDs into disjoint train and validation sets."""
    unique_ids = np.array(sorted(list(set(s1_ids))))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_ids))
    split_idx = int(len(unique_ids) * train_ratio)

    train_ids = set(unique_ids[perm[:split_idx]])
    val_ids = set(unique_ids[perm[split_idx:]])

    return train_ids, val_ids
