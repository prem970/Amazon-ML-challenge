"""
Unicode normalization and text cleaning utilities.
Handles multi-lingual text including US, India, France (accents, diacritics).
"""

import unicodedata
import re

WHITESPACE_RE = re.compile(r'\s+')
NON_ALPHANUMERIC_RE = re.compile(r'[^a-z0-9\s]')
ACCENT_STRIP_RE = re.compile(r'[\u0300-\u036f]')

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters: NFKD decomposition, strip accents, lowercase."""
    if not text or not isinstance(text, str):
        return ""
    # NFKD decomposition decomposes combined characters into base + combining mark
    decomposed = unicodedata.normalize('NFKD', text)
    # Strip combining diacritical marks (e.g. é -> e, à -> a, ü -> u)
    stripped = ACCENT_STRIP_RE.sub('', decomposed)
    return stripped.lower().strip()

def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space."""
    if not text:
        return ""
    return WHITESPACE_RE.sub(' ', text).strip()

def clean_text(text: str) -> str:
    """Standard cleaning pipeline: unicode normalize, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    norm = normalize_unicode(text)
    norm = norm.replace('&', ' and ')
    norm = NON_ALPHANUMERIC_RE.sub(' ', norm)
    return collapse_whitespace(norm)
