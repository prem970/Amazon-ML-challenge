"""
Cross-field interaction, country, and source indicator features.
"""

from typing import Dict, Any

def compute_interaction_features(
    name_feats: Dict[str, float],
    addr_feats: Dict[str, float],
    s1_country: str,
    cand_country: str,
    cand_id: str,
) -> Dict[str, float]:
    """Calculate cross-field interaction terms and country/source indicators."""
    name_sim = name_feats.get("name_jw_legal", 0.0)
    addr_sim = addr_feats.get("addr_jw", 0.0)

    # Cross-field metrics
    name_addr_product = name_sim * addr_sim
    name_addr_mean = (name_sim + addr_sim) / 2.0
    name_addr_min = min(name_sim, addr_sim)
    name_addr_max = max(name_sim, addr_sim)
    name_addr_diff = abs(name_sim - addr_sim)

    # Agreement score
    strong_name = 1.0 if name_sim >= 0.85 else 0.0
    strong_addr = 1.0 if addr_sim >= 0.80 else 0.0
    agreements = (
        strong_name
        + strong_addr
        + addr_feats.get("postal_exact", 0.0)
        + addr_feats.get("house_exact", 0.0)
        + addr_feats.get("state_exact", 0.0)
    )

    # Country features (Open-set compliant)
    c1 = (s1_country or "").strip().lower()
    c2 = (cand_country or "").strip().lower()
    country_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    country_mismatch = 1.0 if (c1 and c2 and c1 != c2) else 0.0
    country_missing = 1.0 if (not c1 or not c2) else 0.0

    # Source indicator
    # cand_id starts with S2- or S3-
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    # Special corner-case indicators
    is_empty_addr_high_name = 1.0 if (addr_feats.get("addr_cand_empty", 0.0) == 1.0 and name_sim >= 0.90) else 0.0
    is_weak_name_high_addr = 1.0 if (name_sim < 0.45 and addr_sim >= 0.85) else 0.0

    return {
        "name_addr_product": name_addr_product,
        "name_addr_mean": name_addr_mean,
        "name_addr_min": name_addr_min,
        "name_addr_max": name_addr_max,
        "name_addr_diff": name_addr_diff,
        "agreement_count": float(agreements),
        "country_exact": country_exact,
        "country_mismatch": country_mismatch,
        "country_missing": country_missing,
        "is_source2": is_source2,
        "is_source3": is_source3,
        "is_empty_addr_high_name": is_empty_addr_high_name,
        "is_weak_name_high_addr": is_weak_name_high_addr,
    }
