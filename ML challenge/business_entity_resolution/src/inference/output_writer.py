"""
Submission file writer and official validation runner.
Guarantees byte-level compliance with Amazon ML Challenge 2026 submission rules.
"""

from typing import Dict, List, Set
from pathlib import Path
import subprocess
import sys
try:
    from ..config import DELIMITER, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, TEST_DIR
except (ImportError, ValueError):
    from config import DELIMITER, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, TEST_DIR

MATCHING_HEADER = f"source1_entity_id{DELIMITER}matched_entity_ids\n"
CANDIDATE_HEADER = f"source1_entity_id{DELIMITER}candidate_entity_ids\n"

def write_submission_files(
    all_test_s1_ids: List[str],
    matching_dict: Dict[str, List[str]],
    candidate_dict: Dict[str, List[str]],
    matching_path: Path = MATCHING_RESULTS_PATH,
    candidate_path: Path = CANDIDATE_PAIRS_PATH,
):
    """Write matching_results.tsv and candidate_pairs.tsv ensuring strictly valid formatting."""
    matching_path = Path(matching_path)
    candidate_path = Path(candidate_path)
    matching_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(MATCHING_HEADER)
        for s1_id in all_test_s1_ids:
            matches = matching_dict.get(s1_id, [])
            # Deduplicate preserving order
            seen = set()
            clean_matches = []
            for mid in matches:
                mid_clean = mid.strip()
                if mid_clean and mid_clean not in seen and not mid_clean.startswith("S1-"):
                    seen.add(mid_clean)
                    clean_matches.append(mid_clean)
            m_str = ",".join(clean_matches)
            f.write(f"{s1_id}{DELIMITER}{m_str}\n")

    # 2. Write candidate_pairs.tsv
    with open(candidate_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(CANDIDATE_HEADER)
        for s1_id in all_test_s1_ids:
            cands = candidate_dict.get(s1_id, [])
            seen = set()
            clean_cands = []
            for cid in cands:
                cid_clean = cid.strip()
                if cid_clean and cid_clean not in seen and not cid_clean.startswith("S1-"):
                    seen.add(cid_clean)
                    clean_cands.append(cid_clean)
            c_str = ",".join(clean_cands)
            f.write(f"{s1_id}{DELIMITER}{c_str}\n")

    print(f"Successfully generated:\n  - {matching_path}\n  - {candidate_path}")

def run_official_validator(
    matching_path: Path = MATCHING_RESULTS_PATH,
    candidate_path: Path = CANDIDATE_PAIRS_PATH,
    test_dir: Path = TEST_DIR,
) -> bool:
    """Execute student_resource/utils/validate_submission.py to verify format."""
    validator_script = test_dir.parent.parent / "utils" / "validate_submission.py"
    if not validator_script.exists():
        validator_script = test_dir.parent / "utils" / "validate_submission.py"
    if not validator_script.exists():
        print(f"Warning: Validator script not found at {validator_script}")
        return False

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]

    print("Executing official validation script...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)
    return res.returncode == 0
