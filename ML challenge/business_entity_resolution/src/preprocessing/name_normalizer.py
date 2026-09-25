"""
Multi-view business name normalization.
Supports raw, basic, legal-normalized, token, compact, and character n-gram views.
Covers international corporate forms across US, India, France.
"""

import re
from typing import Dict, List, Set, Any
from .unicode_utils import normalize_unicode, collapse_whitespace

# Legal forms to normalize or strip
# Ordered by length to match multi-word phrases first
LEGAL_SUFFIXES = [
    r'\bprivate\s+limited\b',
    r'\bpvt\s+ltd\b',
    r'\bpvt\s+limited\b',
    r'\bpublic\s+limited\b',
    r'\bltd\b',
    r'\blimited\b',
    r'\bincorporated\b',
    r'\binc\b',
    r'\bcorporation\b',
    r'\bcorp\b',
    r'\blimited\s+liability\s+company\b',
    r'\blimited\s+liability\s+partnership\b',
    r'\bllc\b',
    r'\bllp\b',
    r'\bcompany\b',
    r'\bco\b',
    r'\bgroup\b',
    r'\benterprises\b',
    r'\benterprise\b',
    r'\bservices\b',
    r'\bservice\b',
    r'\bsolutions\b',
    r'\bsolution\b',
    r'\bholdings\b',
    r'\bholding\b',
    r'\bindustries\b',
    r'\bindustry\b',
    # French corporate forms
    r'\bsas\b',
    r'\bsarl\b',
    r'\bsa\b',
    r'\bsnc\b',
    r'\bsci\b',
    r'\beurl\b',
    # German/European forms
    r'\bgmbh\b',
    r'\bag\b',
]

LEGAL_PATTERN = re.compile(r'(' + '|'.join(LEGAL_SUFFIXES) + r')', re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r'\b(?:www\.)?([a-z0-9\-]+)\.(?:com|org|net|in|co|io|fr|gov|edu)\b', re.IGNORECASE)
CLEAN_PUNCT_PATTERN = re.compile(r'[^a-z0-9\s]')
NON_ALPHANUM_ONLY = re.compile(r'[^a-z0-9]')

class NameNormalizer:
    def __init__(self):
        pass

    def basic_view(self, text: str) -> str:
        """Unicode clean, '&' -> 'and', strip punctuation, collapse whitespace."""
        if not text:
            return ""
        norm = normalize_unicode(text)
        norm = norm.replace('&', ' and ')
        # Remove domain extensions (e.g., example.com -> example)
        norm = DOMAIN_PATTERN.sub(r'\1', norm)
        norm = CLEAN_PUNCT_PATTERN.sub(' ', norm)
        return collapse_whitespace(norm)

    def legal_view(self, basic_text: str) -> str:
        """Remove common legal entity designations and trailing generic corporate noise."""
        if not basic_text:
            return ""
        # Strip legal phrases
        cleaned = LEGAL_PATTERN.sub(' ', basic_text)
        cleaned = collapse_whitespace(cleaned)
        # If aggressive stripping left nothing, retain basic_text
        return cleaned if cleaned else basic_text

    def compact_view(self, text: str) -> str:
        """Alphanumeric only representation with no spaces."""
        if not text:
            return ""
        return NON_ALPHANUM_ONLY.sub('', normalize_unicode(text))

    def token_view(self, text: str) -> List[str]:
        """List of tokens."""
        if not text:
            return []
        return [tok for tok in text.split(' ') if tok]

    def char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams from compact representation."""
        compact = self.compact_view(text)
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i:i+n] for i in range(len(compact) - n + 1)}

    def normalize(self, raw_name: str) -> Dict[str, Any]:
        """Produce all multi-view normalization representations for a business name."""
        raw = raw_name if isinstance(raw_name, str) else ""
        basic = self.basic_view(raw)
        legal = self.legal_view(basic)
        compact = self.compact_view(legal if legal else basic)
        tokens = self.token_view(legal if legal else basic)
        ngrams = self.char_ngrams(legal if legal else basic, n=3)

        return {
            "raw": raw,
            "basic": basic,
            "legal": legal,
            "compact": compact,
            "tokens": tokens,
            "ngrams": ngrams,
        }
