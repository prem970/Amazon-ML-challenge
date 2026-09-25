"""
Candidate generation recall audit module.
"""

from typing import Dict, List, Set

def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, List[str]]
) -> Dict[str, float]:
    """Computes blocking recall ceiling: how many true positive pairs made it into the candidate set."""
    total_true = 0
    retained_true = 0
    candidate_counts = []

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        total_true += len(true_set)
        cand_set = set(candidates.get(s1_id, []))
        candidate_counts.append(len(cand_set))
        retained_true += len(true_set & cand_set)

    recall = (retained_true / total_true) if total_true > 0 else 1.0
    avg_cands = (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0

    return {
        "blocking_recall": round(recall, 5),
        "total_true_matches": total_true,
        "retained_true_matches": retained_true,
        "missed_true_matches": total_true - retained_true,
        "avg_candidates_per_entity": round(avg_cands, 2),
    }
