"""
Diagnostic error analysis module.
Identifies false positives (false merges) and false negatives (missed matches).
"""

from typing import Dict, List, Set, Tuple, Any

def analyze_prediction_errors(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    max_examples: int = 15,
) -> Dict[str, Any]:
    """Categorize prediction errors and report sample offending entities."""
    false_positives = []
    false_negatives = []
    singleton_errors = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton false merge
        if not true_set and pred_set:
            singleton_errors.append({
                "source1_id": s1_id,
                "wrongly_predicted": list(pred_set),
            })
            continue

        # False positives (predicted but not true)
        fps = pred_set - true_set
        if fps:
            false_positives.append({
                "source1_id": s1_id,
                "false_matches": list(fps),
            })

        # False negatives (true but not predicted)
        fns = true_set - pred_set
        if fns:
            false_negatives.append({
                "source1_id": s1_id,
                "missed_matches": list(fns),
            })

    return {
        "num_false_positive_entities": len(false_positives),
        "num_false_negative_entities": len(false_negatives),
        "num_singleton_errors": len(singleton_errors),
        "sample_false_positives": false_positives[:max_examples],
        "sample_false_negatives": false_negatives[:max_examples],
        "sample_singleton_errors": singleton_errors[:max_examples],
    }
