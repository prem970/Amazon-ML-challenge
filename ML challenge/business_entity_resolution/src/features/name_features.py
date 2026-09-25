"""
Pairwise name feature extraction across multi-view normalizations.
"""

from typing import Dict, Any
from .string_similarity import (
    levenshtein_ratio,
    token_sort_ratio,
    token_set_ratio,
    jaro_winkler_sim,
    jaccard_similarity,
    ngram_jaccard,
    prefix_match_ratio,
)

def compute_name_features(s1_name: Dict[str, Any], cand_name: Dict[str, Any]) -> Dict[str, float]:
    """Compute rich pairwise features between Source 1 and candidate name multi-views."""
    raw1, raw2 = s1_name.get("raw", ""), cand_name.get("raw", "")
    basic1, basic2 = s1_name.get("basic", ""), cand_name.get("basic", "")
    legal1, legal2 = s1_name.get("legal", ""), cand_name.get("legal", "")
    compact1, compact2 = s1_name.get("compact", ""), cand_name.get("compact", "")
    tokens1, tokens2 = s1_name.get("tokens", []), cand_name.get("tokens", [])
    ngrams1, ngrams2 = s1_name.get("ngrams", set()), cand_name.get("ngrams", set())

    # Exact matches
    exact_raw = 1.0 if (raw1 and raw1 == raw2) else 0.0
    exact_basic = 1.0 if (basic1 and basic1 == basic2) else 0.0
    exact_legal = 1.0 if (legal1 and legal1 == legal2) else 0.0
    exact_compact = 1.0 if (compact1 and compact1 == compact2) else 0.0

    # Multi-view similarities
    sim_lev_basic = levenshtein_ratio(basic1, basic2)
    sim_lev_legal = levenshtein_ratio(legal1, legal2)
    sim_jw_basic = jaro_winkler_sim(basic1, basic2)
    sim_jw_legal = jaro_winkler_sim(legal1, legal2)
    sim_token_sort = token_sort_ratio(basic1, basic2)
    sim_token_set = token_set_ratio(basic1, basic2)
    sim_jaccard_tokens = jaccard_similarity(tokens1, tokens2)
    sim_ngram = ngram_jaccard(ngrams1, ngrams2)
    sim_prefix = prefix_match_ratio(compact1, compact2)

    # First & Last token matches
    first_tok_match = 1.0 if (tokens1 and tokens2 and tokens1[0] == tokens2[0]) else 0.0
    last_tok_match = 1.0 if (tokens1 and tokens2 and tokens1[-1] == tokens2[-1]) else 0.0

    # Length and token counts
    len1, len2 = len(basic1), len(basic2)
    len_diff = abs(len1 - len2)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0
    tok_diff = abs(len(tokens1) - len(tokens2))

    return {
        "name_exact_raw": exact_raw,
        "name_exact_basic": exact_basic,
        "name_exact_legal": exact_legal,
        "name_exact_compact": exact_compact,
        "name_lev_basic": sim_lev_basic,
        "name_lev_legal": sim_lev_legal,
        "name_jw_basic": sim_jw_basic,
        "name_jw_legal": sim_jw_legal,
        "name_token_sort": sim_token_sort,
        "name_token_set": sim_token_set,
        "name_jaccard_tokens": sim_jaccard_tokens,
        "name_ngram_jaccard": sim_ngram,
        "name_prefix_sim": sim_prefix,
        "name_first_token_match": first_tok_match,
        "name_last_token_match": last_tok_match,
        "name_len_diff": float(len_diff),
        "name_len_ratio": sim_jw_legal if (len1 == 0 or len2 == 0) else len_ratio,
        "name_tok_diff": float(tok_diff),
    }
