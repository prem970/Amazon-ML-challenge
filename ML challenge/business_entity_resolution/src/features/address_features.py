"""
Pairwise address and structured component feature extraction.
"""

from typing import Dict, Any
from .string_similarity import (
    levenshtein_ratio,
    token_set_ratio,
    jaro_winkler_sim,
    jaccard_similarity,
    ngram_jaccard,
)

def compute_address_features(
    s1_addr: Dict[str, Any],
    cand_addr: Dict[str, Any],
    s1_comp: Dict[str, Any],
    cand_comp: Dict[str, Any]
) -> Dict[str, float]:
    """Compute rich pairwise features between Source 1 and candidate address views and components."""
    basic1, basic2 = s1_addr.get("basic", ""), cand_addr.get("basic", "")
    tokens1, tokens2 = s1_addr.get("tokens", []), cand_addr.get("tokens", [])
    ngrams1, ngrams2 = s1_addr.get("ngrams", set()), cand_addr.get("ngrams", set())

    # Missing indicators
    s1_empty = 1.0 if not basic1 else 0.0
    cand_empty = 1.0 if not basic2 else 0.0
    both_empty = 1.0 if (s1_empty and cand_empty) else 0.0
    one_empty = 1.0 if (s1_empty != cand_empty) else 0.0

    # Address similarities (if both present, else 0.0)
    if not s1_empty and not cand_empty:
        addr_exact = 1.0 if basic1 == basic2 else 0.0
        addr_lev = levenshtein_ratio(basic1, basic2)
        addr_jw = jaro_winkler_sim(basic1, basic2)
        addr_token_set = token_set_ratio(basic1, basic2)
        addr_jaccard = jaccard_similarity(tokens1, tokens2)
        addr_ngram = ngram_jaccard(ngrams1, ngrams2)
    else:
        addr_exact = 0.0
        addr_lev = 0.0
        addr_jw = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0
        addr_ngram = 0.0

    # Structured component matches
    post1, post2 = s1_comp.get("postal_code"), cand_comp.get("postal_code")
    postal_exact = 1.0 if (post1 and post2 and post1 == post2) else 0.0
    postal_mismatch = 1.0 if (post1 and post2 and post1 != post2) else 0.0

    house1, house2 = s1_comp.get("house_number"), cand_comp.get("house_number")
    house_exact = 1.0 if (house1 and house2 and house1 == house2) else 0.0
    house_mismatch = 1.0 if (house1 and house2 and house1 != house2) else 0.0

    state1, state2 = s1_comp.get("state"), cand_comp.get("state")
    state_exact = 1.0 if (state1 and state2 and state1 == state2) else 0.0
    state_mismatch = 1.0 if (state1 and state2 and state1 != state2) else 0.0

    comp_agreements = postal_exact + house_exact + state_exact
    comp_conflicts = postal_mismatch + house_mismatch + state_mismatch

    return {
        "addr_s1_empty": s1_empty,
        "addr_cand_empty": cand_empty,
        "addr_both_empty": both_empty,
        "addr_one_empty": one_empty,
        "addr_exact": addr_exact,
        "addr_lev": addr_lev,
        "addr_jw": addr_jw,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "addr_ngram": addr_ngram,
        "postal_exact": postal_exact,
        "postal_mismatch": postal_mismatch,
        "house_exact": house_exact,
        "house_mismatch": house_mismatch,
        "state_exact": state_exact,
        "state_mismatch": state_mismatch,
        "comp_agreements": float(comp_agreements),
        "comp_conflicts": float(comp_conflicts),
    }
