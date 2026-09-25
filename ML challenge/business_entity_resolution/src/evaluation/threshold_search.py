"""
Joint (alpha, tau) hyperparameter grid search for the XGBoost + LightGBM hybrid model.
Directly maximizes validation Macro F_0.5.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
from .metrics import compute_macro_f05
try:
    from ..config import HYBRID_CONFIG
except (ImportError, ValueError):
    from config import HYBRID_CONFIG

def search_optimal_alpha_threshold(
    val_pairs: List[Tuple[str, str]],  # (s1_id, cand_id)
    xgb_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
) -> Dict[str, Any]:
    """Perform joint coarse and fine grid search to find optimal alpha and threshold tau."""
    # Pre-organize pairs by s1_id for fast prediction construction
    s1_to_indices = {}
    for idx, (s1_id, _) in enumerate(val_pairs):
        if s1_id not in s1_to_indices:
            s1_to_indices[s1_id] = []
        s1_to_indices[s1_id].append(idx)

    # Coarse search
    alphas = HYBRID_CONFIG["alpha_coarse"]
    thresholds = np.arange(
        HYBRID_CONFIG["threshold_coarse_min"],
        HYBRID_CONFIG["threshold_coarse_max"] + 0.001,
        HYBRID_CONFIG["threshold_coarse_step"]
    )

    best_score = -1.0
    best_alpha = 0.5
    best_tau = 0.70

    for alpha in alphas:
        p_hybrid = alpha * xgb_probs + (1.0 - alpha) * lgbm_probs

        for tau in thresholds:
            # Build entity-level predictions
            preds: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_val_s1_ids}
            for s1_id, indices in s1_to_indices.items():
                for i in indices:
                    if p_hybrid[i] >= tau:
                        preds[s1_id].add(val_pairs[i][1])

            res = compute_macro_f05(ground_truth, preds)
            score = res["macro_f05"]
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_tau = tau

    # Fine search around best coarse (alpha, tau)
    fine_alphas = [max(0.0, best_alpha - 0.1), best_alpha, min(1.0, best_alpha + 0.1)]
    fine_taus = np.arange(
        max(0.40, best_tau - 0.04),
        min(0.99, best_tau + 0.04) + 0.001,
        HYBRID_CONFIG["threshold_fine_step"]
    )

    for alpha in fine_alphas:
        p_hybrid = alpha * xgb_probs + (1.0 - alpha) * lgbm_probs
        for tau in fine_taus:
            preds = {s1_id: set() for s1_id in all_val_s1_ids}
            for s1_id, indices in s1_to_indices.items():
                for i in indices:
                    if p_hybrid[i] >= tau:
                        preds[s1_id].add(val_pairs[i][1])

            res = compute_macro_f05(ground_truth, preds)
            score = res["macro_f05"]
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_tau = tau

    return {
        "best_alpha": round(best_alpha, 4),
        "best_threshold": round(best_tau, 4),
        "best_val_macro_f05": round(best_score, 5),
    }
