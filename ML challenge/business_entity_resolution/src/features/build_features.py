"""
Feature builder pipeline.
Generates an ordered numerical feature vector for any candidate pair (Source 1, Candidate).
"""

from typing import Dict, List, Any
import numpy as np
from .name_features import compute_name_features
from .address_features import compute_address_features
from .interaction_features import compute_interaction_features

FEATURE_NAMES = [
    # Name Features (18)
    "name_exact_raw",
    "name_exact_basic",
    "name_exact_legal",
    "name_exact_compact",
    "name_lev_basic",
    "name_lev_legal",
    "name_jw_basic",
    "name_jw_legal",
    "name_token_sort",
    "name_token_set",
    "name_jaccard_tokens",
    "name_ngram_jaccard",
    "name_prefix_sim",
    "name_first_token_match",
    "name_last_token_match",
    "name_len_diff",
    "name_len_ratio",
    "name_tok_diff",
    # Address Features (17)
    "addr_s1_empty",
    "addr_cand_empty",
    "addr_both_empty",
    "addr_one_empty",
    "addr_exact",
    "addr_lev",
    "addr_jw",
    "addr_token_set",
    "addr_jaccard",
    "addr_ngram",
    "postal_exact",
    "postal_mismatch",
    "house_exact",
    "house_mismatch",
    "state_exact",
    "state_mismatch",
    "comp_agreements",
    "comp_conflicts",
    # Interaction & Context Features (13)
    "name_addr_product",
    "name_addr_mean",
    "name_addr_min",
    "name_addr_max",
    "name_addr_diff",
    "agreement_count",
    "country_exact",
    "country_mismatch",
    "country_missing",
    "is_source2",
    "is_source3",
    "is_empty_addr_high_name",
    "is_weak_name_high_addr",
]

def extract_pairwise_feature_vector(
    s1_name: Dict[str, Any],
    cand_name: Dict[str, Any],
    s1_addr: Dict[str, Any],
    cand_addr: Dict[str, Any],
    s1_comp: Dict[str, Any],
    cand_comp: Dict[str, Any],
    s1_country: str,
    cand_country: str,
    cand_id: str,
) -> np.ndarray:
    """Extract a 1D float32 numpy array matching FEATURE_NAMES exactly."""
    name_feats = compute_name_features(s1_name, cand_name)
    addr_feats = compute_address_features(s1_addr, cand_addr, s1_comp, cand_comp)
    inter_feats = compute_interaction_features(name_feats, addr_feats, s1_country, cand_country, cand_id)

    combined = {**name_feats, **addr_feats, **inter_feats}
    return np.array([combined[k] for k in FEATURE_NAMES], dtype=np.float32)
