"""
Address text normalizer and standardizer.
Handles international road/street abbreviations, unit markers, landmarks across US, India, France.
"""

import re
from typing import Dict, List, Set, Any
from .unicode_utils import normalize_unicode, collapse_whitespace

# Common address token expansions
ADDRESS_ABBR_MAP = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bbvd\b': 'boulevard',
    r'\bbd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bterr?\b': 'terrace',
    r'\baly\b': 'alley',
    r'\bway\b': 'way',
    r'\bflr?\b': 'floor',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bbldg\b': 'building',
    r'\bopp\b': 'opposite',
    r'\bnr\b': 'near',
    r'\bb/h\b': 'behind',
    r'\bbh\b': 'behind',
    # French
    r'\brte\b': 'route',
    r'\ball\b': 'allee',
    r'\bimp\b': 'impasse',
    r'\bchem\b': 'chemin',
}

COMPILED_ABBRS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR_MAP.items()]
CLEAN_PUNCT_PATTERN = re.compile(r'[^a-z0-9\s]')
NON_ALPHANUM_ONLY = re.compile(r'[^a-z0-9]')

class AddressNormalizer:
    def __init__(self):
        pass

    def basic_view(self, text: str) -> str:
        """Unicode clean, expand standard road/street abbreviations, strip punctuation."""
        if not text:
            return ""
        norm = normalize_unicode(text)
        for pattern, repl in COMPILED_ABBRS:
            norm = pattern.sub(repl, norm)
        norm = CLEAN_PUNCT_PATTERN.sub(' ', norm)
        return collapse_whitespace(norm)

    def compact_view(self, text: str) -> str:
        """Alphanumeric compact representation."""
        if not text:
            return ""
        return NON_ALPHANUM_ONLY.sub('', normalize_unicode(text))

    def token_view(self, basic_text: str) -> List[str]:
        if not basic_text:
            return []
        return [tok for tok in basic_text.split(' ') if tok]

    def char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        compact = self.compact_view(text)
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i:i+n] for i in range(len(compact) - n + 1)}

    def normalize(self, raw_addr: str) -> Dict[str, Any]:
        raw = raw_addr if isinstance(raw_addr, str) else ""
        basic = self.basic_view(raw)
        compact = self.compact_view(basic)
        tokens = self.token_view(basic)
        ngrams = self.char_ngrams(basic, n=3)

        return {
            "raw": raw,
            "basic": basic,
            "compact": compact,
            "tokens": tokens,
            "ngrams": ngrams,
        }
