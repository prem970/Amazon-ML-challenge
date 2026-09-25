"""
Structured address component extraction.
Extracts Postal/PIN/ZIP codes, house/building numbers, city/locality, and state.
Designed for open-set countries with specific rules for US, India, France.
"""

import re
from typing import Dict, Optional, Any
from .unicode_utils import normalize_unicode, collapse_whitespace

# Postal code patterns
US_ZIP_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
INDIA_PIN_RE = re.compile(r'\b([1-9]\d{5})\b')
FRANCE_POSTAL_RE = re.compile(r'\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b')

# House / Building / Flat number patterns
HOUSE_NUM_RE = re.compile(
    r'(?:^|[,\s])(?:h\.?no\.?|house\s+no\.?|flat\s+no\.?|plot\s+no\.?|unit|suite|#)\s*([a-z0-9\-\/]+)'
    r'|(?:^|[,\s])([0-9]{1,6}[a-z]?(?:[\-\/][0-9]{1,6}[a-z]?)?)(?:\s+[a-z]+)',
    re.IGNORECASE
)

# US States (codes and full names)
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri',
    'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey',
    'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio',
    'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming',
    'dc': 'district of columbia'
}
US_STATE_PATTERN = re.compile(r'\b(' + '|'.join(US_STATES.keys()) + r')\b|\b(' + '|'.join(US_STATES.values()) + r')\b', re.IGNORECASE)

# Indian States
INDIA_STATES = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
    'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
    'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
    'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh',
    'uttarakhand', 'west bengal', 'delhi', 'jammu and kashmir', 'ladakh', 'chandigarh',
    'puducherry'
]
INDIA_STATE_PATTERN = re.compile(r'\b(' + '|'.join(INDIA_STATES) + r')\b', re.IGNORECASE)

class AddressComponentExtractor:
    def __init__(self):
        pass

    def extract_postal_code(self, raw_addr: str, country: str = "") -> Optional[str]:
        """Extract postal/ZIP/PIN code based on country or general pattern."""
        if not raw_addr:
            return None
        text = raw_addr
        cntry = country.strip().upper() if country else ""

        if cntry == "INDIA":
            m = INDIA_PIN_RE.search(text)
            if m:
                return m.group(1)
        elif cntry == "US":
            m = US_ZIP_RE.search(text)
            if m:
                return m.group(1)
        elif cntry == "FRANCE":
            m = FRANCE_POSTAL_RE.search(text)
            if m:
                return m.group(1)
        else:
            # Fallback across patterns
            m = INDIA_PIN_RE.search(text) or US_ZIP_RE.search(text) or FRANCE_POSTAL_RE.search(text)
            if m:
                return m.group(1)
        return None

    def extract_house_number(self, raw_addr: str) -> Optional[str]:
        """Extract building, plot, house or street number."""
        if not raw_addr:
            return None
        m = HOUSE_NUM_RE.search(raw_addr)
        if m:
            val = m.group(1) or m.group(2)
            if val:
                val = val.strip().lower()
                # filter out pure non-digits if too long
                if any(c.isdigit() for c in val):
                    return val
        return None

    def extract_state(self, raw_addr: str, country: str = "") -> Optional[str]:
        """Extract standard state name or code."""
        if not raw_addr:
            return None
        norm = normalize_unicode(raw_addr)
        cntry = country.strip().upper() if country else ""

        if cntry == "US" or not cntry:
            m = US_STATE_PATTERN.search(norm)
            if m:
                code_or_name = (m.group(1) or m.group(2)).lower()
                return US_STATES.get(code_or_name, code_or_name)

        if cntry == "INDIA" or not cntry:
            m = INDIA_STATE_PATTERN.search(norm)
            if m:
                return m.group(1).lower()

        return None

    def extract_components(self, raw_addr: str, country: str = "") -> Dict[str, Any]:
        postal = self.extract_postal_code(raw_addr, country)
        house_num = self.extract_house_number(raw_addr)
        state = self.extract_state(raw_addr, country)

        return {
            "postal_code": postal,
            "house_number": house_num,
            "state": state,
            "has_postal": postal is not None,
            "has_house_num": house_num is not None,
            "has_state": state is not None,
        }
