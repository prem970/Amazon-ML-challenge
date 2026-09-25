"""
Ultra-fast streaming inference runner.
Uses SQLite zero-RAM indexing, vectorized batch model scoring, and direct line I/O.
Guarantees byte-level submission compliance with Amazon ML Challenge 2026 rules.
"""

import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np

try:
    from ..config import (
        TEST_SOURCE1_PATH,
        TEST_SOURCE2_PATH,
        TEST_SOURCE3_PATH,
        MATCHING_RESULTS_PATH,
        CANDIDATE_PAIRS_PATH,
        MODELS_DIR,
        BLOCKING_CONFIG,
    )
    from ..preprocessing import NameNormalizer, AddressNormalizer, AddressComponentExtractor
    from ..features.build_features import extract_pairwise_feature_vector
    from .safety_layer import PrecisionSafetyLayer
    from .output_writer import run_official_validator
except (ImportError, ValueError):
    from config import (
        TEST_SOURCE1_PATH,
        TEST_SOURCE2_PATH,
        TEST_SOURCE3_PATH,
        MATCHING_RESULTS_PATH,
        CANDIDATE_PAIRS_PATH,
        MODELS_DIR,
        BLOCKING_CONFIG,
    )
    from preprocessing import NameNormalizer, AddressNormalizer, AddressComponentExtractor
    from features.build_features import extract_pairwise_feature_vector
    from inference.safety_layer import PrecisionSafetyLayer
    from inference.output_writer import run_official_validator

def build_target_database(db_path: Path, max_targets_per_source: Optional[int] = None) -> sqlite3.Connection:
    """Index test Source 2 and Source 3 records into a local SQLite database for zero-RAM lookup."""
    db_exists = db_path.exists()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA cache_size = 50000;")

    if db_exists:
        try:
            cur.execute("SELECT count(*) FROM target_records;")
            cnt = cur.fetchone()[0]
            if cnt > 1000:
                print(f"Target database already indexed ({cnt} records). Reusing existing index.")
                sys.stdout.flush()
                return conn
        except Exception:
            pass

    cur.execute("""
    CREATE TABLE IF NOT EXISTS target_records (
        entity_id TEXT PRIMARY KEY,
        legal_name TEXT,
        compact_name TEXT,
        basic_addr TEXT,
        country TEXT,
        postal_code TEXT,
        house_num TEXT,
        state TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS block_index (
        block_key TEXT,
        entity_id TEXT
    );
    """)

    name_norm = NameNormalizer()
    addr_norm = AddressNormalizer()
    addr_comp = AddressComponentExtractor()

    batch_targets = []
    batch_blocks = []
    batch_size = 25000
    total_indexed = 0

    print("Building high-speed target index from test sources...")
    sys.stdout.flush()

    for source_file in [TEST_SOURCE2_PATH, TEST_SOURCE3_PATH]:
        if not source_file.exists():
            continue
        print(f"Streaming and indexing {source_file.name}...")
        sys.stdout.flush()
        count = 0

        with open(source_file, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0].strip():
                    continue
                eid = parts[0].strip()
                raw_name = parts[1].strip() if len(parts) > 1 else ""
                raw_addr = parts[2].strip() if len(parts) > 2 else ""
                country = parts[3].strip() if len(parts) > 3 else ""

                n_dict = name_norm.normalize(raw_name)
                a_dict = addr_norm.normalize(raw_addr)
                c_dict = addr_comp.extract_components(raw_addr, country)

                legal = n_dict.get("legal", "")
                compact = n_dict.get("compact", "")
                basic_addr = a_dict.get("basic", "")
                postal = c_dict.get("postal_code") or ""
                house = c_dict.get("house_number") or ""
                state = c_dict.get("state") or ""

                batch_targets.append((eid, legal, compact, basic_addr, country, postal, house, state))

                # Blocking keys
                if legal and len(legal) >= 3:
                    batch_blocks.append((f"n_{legal}", eid))
                if compact and len(compact) >= 8:
                    batch_blocks.append((f"p_{compact[:8]}", eid))
                tokens = n_dict.get("tokens", [])
                if tokens:
                    longest_tok = max(tokens, key=len)
                    if len(longest_tok) >= 5:
                        batch_blocks.append((f"t_{longest_tok}", eid))
                if postal:
                    batch_blocks.append((f"z_{postal}", eid))
                if state and tokens:
                    batch_blocks.append((f"sn_{state}_{tokens[0]}", eid))

                count += 1
                if len(batch_targets) >= batch_size:
                    cur.executemany("INSERT OR IGNORE INTO target_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch_targets)
                    cur.executemany("INSERT INTO block_index VALUES (?, ?)", batch_blocks)
                    conn.commit()
                    total_indexed += len(batch_targets)
                    batch_targets.clear()
                    batch_blocks.clear()

                if max_targets_per_source and count >= max_targets_per_source:
                    break

    if batch_targets:
        cur.executemany("INSERT OR IGNORE INTO target_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch_targets)
        cur.executemany("INSERT INTO block_index VALUES (?, ?)", batch_blocks)
        conn.commit()
        total_indexed += len(batch_targets)

    print(f"Creating B-Tree index over blocking keys ({total_indexed} target records)...")
    sys.stdout.flush()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bkey ON block_index(block_key);")
    conn.commit()
    print("Target indexing complete!")
    sys.stdout.flush()
    return conn

def run_test_inference(
    hybrid_model,
    max_s1: Optional[int] = None,
    output_matching_path: Path = MATCHING_RESULTS_PATH,
    output_candidate_path: Path = CANDIDATE_PAIRS_PATH,
    db_path: Optional[Path] = None,
):
    """
    Execute vectorized streaming inference:
    - Queries target SQLite index for candidate matches
    - Extracts features in vectorized arrays
    - Vectorized model scoring
    - Precision safety guarding
    - Direct line streaming to disk
    """
    if db_path is None:
        db_path = MODELS_DIR / "test_targets.db"

    # 1. Connect / Build Target Database
    conn = build_target_database(db_path, max_targets_per_source=250000 if max_s1 else None)
    cur = conn.cursor()

    name_norm = NameNormalizer()
    addr_norm = AddressNormalizer()
    addr_comp = AddressComponentExtractor()
    safety = PrecisionSafetyLayer(threshold=hybrid_model.threshold)

    output_matching_path = Path(output_matching_path)
    output_candidate_path = Path(output_candidate_path)
    output_matching_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Executing test inference. Writing results to:\n  - {output_matching_path}\n  - {output_candidate_path}")
    sys.stdout.flush()
    t0 = time.time()
    s1_processed = 0

    with open(output_matching_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(output_candidate_path, "w", encoding="utf-8", newline="\n") as f_cand, \
         open(TEST_SOURCE1_PATH, "r", encoding="utf-8", errors="replace") as f_in:

        # Header
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_in, None)  # Skip header

        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()

            # If user specified max_s1 and we exceeded budget, emit empty singleton fast
            if max_s1 and s1_processed >= max_s1:
                f_match.write(f"{s1_id}\t\n")
                f_cand.write(f"{s1_id}\t\n")
                s1_processed += 1
                continue

            raw_name = parts[1].strip() if len(parts) > 1 else ""
            raw_addr = parts[2].strip() if len(parts) > 2 else ""
            country = parts[3].strip() if len(parts) > 3 else ""

            s1_n = name_norm.normalize(raw_name)
            s1_a = addr_norm.normalize(raw_addr)
            s1_c = addr_comp.extract_components(raw_addr, country)

            legal = s1_n.get("legal", "")
            compact = s1_n.get("compact", "")
            tokens = s1_n.get("tokens", [])
            postal = s1_c.get("postal_code") or ""
            state = s1_c.get("state") or ""

            # Query keys
            query_keys = []
            if legal and len(legal) >= 3:
                query_keys.append(f"n_{legal}")
            if compact and len(compact) >= 8:
                query_keys.append(f"p_{compact[:8]}")
            if tokens:
                longest_tok = max(tokens, key=len)
                if len(longest_tok) >= 5:
                    query_keys.append(f"t_{longest_tok}")
            if postal:
                query_keys.append(f"z_{postal}")
            if state and tokens:
                query_keys.append(f"sn_{state}_{tokens[0]}")

            candidates = []
            if query_keys:
                placeholders = ",".join("?" for _ in query_keys)
                cur.execute(
                    f"SELECT DISTINCT entity_id FROM block_index WHERE block_key IN ({placeholders}) LIMIT {BLOCKING_CONFIG['max_candidates_per_entity']};",
                    query_keys
                )
                candidates = [r[0] for r in cur.fetchall()]

            matched_ids = []
            if candidates:
                cand_placeholders = ",".join("?" for _ in candidates)
                cur.execute(
                    f"SELECT entity_id, legal_name, compact_name, basic_addr, country, postal_code, house_num, state FROM target_records WHERE entity_id IN ({cand_placeholders});",
                    candidates
                )
                rows = cur.fetchall()

                if rows:
                    feat_matrix = []
                    cand_meta = []
                    for cid, c_legal, c_compact, c_baddr, c_country, c_post, c_house, c_state in rows:
                        cand_n = {"raw": c_legal, "basic": c_legal, "legal": c_legal, "compact": c_compact, "tokens": c_legal.split(), "ngrams": set()}
                        cand_a = {"raw": c_baddr, "basic": c_baddr, "compact": "", "tokens": c_baddr.split(), "ngrams": set()}
                        cand_comp = {"postal_code": c_post, "house_number": c_house, "state": c_state}

                        vec = extract_pairwise_feature_vector(
                            s1_name=s1_n,
                            cand_name=cand_n,
                            s1_addr=s1_a,
                            cand_addr=cand_a,
                            s1_comp=s1_c,
                            cand_comp=cand_comp,
                            s1_country=country,
                            cand_country=c_country,
                            cand_id=cid,
                        )
                        feat_matrix.append(vec)
                        cand_meta.append((cid, c_country))

                    # Vectorized batch prediction across all candidates of this entity
                    X_entity = np.array(feat_matrix, dtype=np.float32)
                    probs = hybrid_model.predict_proba(X_entity)

                    cand_scores = [
                        (cand_meta[i][0], float(probs[i]), {"s1_country": country, "cand_country": cand_meta[i][1]})
                        for i in range(len(probs))
                    ]
                    matched_ids = safety.filter_entity_matches(cand_scores)

            f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")
            f_cand.write(f"{s1_id}\t{','.join(candidates)}\n")

            s1_processed += 1
            if s1_processed % 50000 == 0:
                elapsed = time.time() - t0
                rate = s1_processed / elapsed
                print(f"Processed {s1_processed:,} Source 1 entities ({rate:.0f} entities/sec)...")
                sys.stdout.flush()

    conn.close()
    print(f"Finished inference on {s1_processed:,} entities in {time.time() - t0:.1f} seconds.")
    sys.stdout.flush()

    # Validate output using official validator
    print("Running official submission validator...")
    sys.stdout.flush()
    is_valid = run_official_validator(
        matching_path=output_matching_path,
        candidate_path=output_candidate_path,
    )
    if is_valid:
        print("PASS: Submission is 100% compliant with competition rules!")
    else:
        print("Note: Review validator feedback above.")
    sys.stdout.flush()
