"""
Data validation module.
Verifies TSV formatting, entity ID prefixes, schema conformity, and data integrity.
"""

from typing import Dict, List, Set, Tuple
from pathlib import Path
try:
    from ..config import S1_PREFIX, S2_PREFIX, S3_PREFIX, DELIMITER
except (ImportError, ValueError):
    from config import S1_PREFIX, S2_PREFIX, S3_PREFIX, DELIMITER

REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

class DataValidator:
    def __init__(self):
        pass

    def validate_source_file_header(self, file_path: Path) -> List[str]:
        errors = []
        if not file_path.exists():
            return [f"File missing: {file_path}"]
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split(DELIMITER)
            header_lower = [c.strip().lower() for c in header]
            if header_lower != REQUIRED_SOURCE_COLS:
                errors.append(f"{file_path.name}: header is {header_lower}, expected {REQUIRED_SOURCE_COLS}")
        return errors

    def validate_ground_truth_header(self, file_path: Path) -> List[str]:
        errors = []
        if not file_path.exists():
            return [f"File missing: {file_path}"]
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split(DELIMITER)
            header_lower = [c.strip().lower() for c in header]
            if header_lower != REQUIRED_GT_COLS:
                errors.append(f"{file_path.name}: header is {header_lower}, expected {REQUIRED_GT_COLS}")
        return errors

    def validate_entity_prefix(self, entity_id: str, expected_prefix: str) -> bool:
        return entity_id.startswith(expected_prefix)

    def validate_matched_ids(self, matched_str: str) -> Tuple[bool, List[str]]:
        """Validate that all IDs in matched_str start with S2- or S3-."""
        if not matched_str or not matched_str.strip():
            return True, []
        invalid = []
        ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        for mid in ids:
            if not (mid.startswith(S2_PREFIX) or mid.startswith(S3_PREFIX)):
                invalid.append(mid)
        return len(invalid) == 0, invalid
