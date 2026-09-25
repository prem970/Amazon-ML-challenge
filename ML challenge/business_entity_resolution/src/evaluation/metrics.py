"""
Evaluation metrics module.
Implements the exact official Macro F_0.5 evaluation metric for the Amazon ML Challenge 2026.
"""

from typing import Dict, Set, List, Tuple
import numpy as np

def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F_0.5 for a single Source 1 entity."""
    # Singleton case: entity has no true matches
    if not true_ids:
        if not pred_ids:
            return 1.0, 1.0, 1.0  # Correctly predicted singleton
        else:
            return 0.0, 1.0, 0.0  # False merge on singleton

    # Entity has true matches but none were predicted
    if not pred_ids:
        return 1.0, 0.0, 0.0

    # Non-empty ground truth and non-empty prediction
    intersection = len(true_ids & pred_ids)
    precision = intersection / len(pred_ids)
    recall = intersection / len(true_ids)

    if precision == 0.0 or recall == 0.0:
        return precision, recall, 0.0

    # Official formula: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05

def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5, Precision, and Recall across all Source 1 entities.
    Every S1 in ground_truth is scored. If missing from predictions, it defaults to empty set.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    precisions = []
    recalls = []
    f05_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f05 = compute_entity_f05(true_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "total_entities": total_entities,
    }
