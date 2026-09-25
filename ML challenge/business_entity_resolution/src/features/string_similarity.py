"""
High-performance string similarity metrics.
Utilizes rapidfuzz (C++ backed) for speed with robust fallbacks.
"""

from typing import Set, List
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def levenshtein_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0
    # Simple fallback
    set1, set2 = set(s1), set(s2)
    return len(set1 & set2) / max(len(set1 | set2), 1)

def token_sort_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    return levenshtein_ratio(' '.join(sorted(s1.split())), ' '.join(sorted(s2.split())))

def token_set_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(s1, s2) / 100.0
    tokens1, tokens2 = set(s1.split()), set(s2.split())
    return len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)

def jaro_winkler_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(s1, s2)
    return levenshtein_ratio(s1, s2)

def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) if union > 0 else 0.0

def ngram_jaccard(ngrams1: Set[str], ngrams2: Set[str]) -> float:
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return (intersection / union) if union > 0 else 0.0

def prefix_match_ratio(s1: str, s2: str, max_len: int = 10) -> float:
    if not s1 or not s2:
        return 0.0
    common = 0
    lim = min(len(s1), len(s2), max_len)
    for i in range(lim):
        if s1[i] == s2[i]:
            common += 1
        else:
            break
    return common / max(min(len(s1), len(s2)), 1)
