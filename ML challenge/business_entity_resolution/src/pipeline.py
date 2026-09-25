"""
End-to-End Pipeline for Business Entity Resolution.
Orchestrates data loading, multi-view preprocessing, blocking, feature extraction,
entity-aware validation, XGBoost + LightGBM training, joint (alpha, threshold) tuning,
and test inference.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
    EXPERIMENTS_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    RANDOM_SEED,
)
from data.loader import stream_tsv_records, load_ground_truth_df
from data.validator import DataValidator
from data.profiling import profile_source_tsv
from preprocessing import NameNormalizer, AddressNormalizer, AddressComponentExtractor
from blocking import MultiPassCandidateGenerator
from features.build_features import extract_pairwise_feature_vector, FEATURE_NAMES
from training import (
    split_source1_entities,
    build_training_pairs,
    mine_hard_negatives,
    train_xgboost_model,
    train_lightgbm_model,
    HybridEntityResolver,
)
from evaluation import (
    compute_macro_f05,
    search_optimal_alpha_threshold,
    evaluate_blocking_recall,
    analyze_prediction_errors,
)
from inference import write_submission_files, run_official_validator, PrecisionSafetyLayer

class EntityResolutionPipeline:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.name_norm = NameNormalizer()
        self.addr_norm = AddressNormalizer()
        self.addr_comp = AddressComponentExtractor()
        self.hybrid_model = None

    def run_training_and_validation(
        self,
        num_s1_records: int = 50000,
        num_target_records: int = 150000,
        enable_hard_negatives: bool = True,
    ) -> Dict[str, any]:
        """
        Train and evaluate the end-to-end hybrid architecture.
        """
        start_time = time.time()
        print("=" * 70)
        print("Starting Business Entity Resolution Training & Validation Pipeline")
        print(f"S1 sample budget: {num_s1_records} | Target sample budget: {num_target_records}")
        print("=" * 70)

        # 1. Load ground truth for the S1 sample
        print("\n[Step 1/8] Loading Ground Truth...")
        gt_map: Dict[str, Set[str]] = {}
        for rec in stream_tsv_records(str(TRAIN_GROUND_TRUTH_PATH)):
            s1_id = rec["source1_entity_id"]
            mids = rec.get("matched_entity_ids", "").strip()
            if mids:
                gt_map[s1_id] = {m.strip() for m in mids.split(",") if m.strip()}
            else:
                gt_map[s1_id] = set()

            if len(gt_map) >= num_s1_records:
                break

        s1_ids_selected = set(gt_map.keys())
        target_ids_needed = set()
        for mids in gt_map.values():
            target_ids_needed.update(mids)

        print(f"Loaded ground truth for {len(gt_map)} Source 1 entities.")
        print(f"Identified {len(target_ids_needed)} distinct true match target records.")

        # 2. Ingest Source 1 records
        print("\n[Step 2/8] Ingesting & Normalizing Source 1 Records...")
        s1_data = {}
        for rec in stream_tsv_records(str(TRAIN_SOURCE1_PATH)):
            eid = rec["entity_id"]
            if eid in s1_ids_selected:
                raw_name = rec.get("business_name", "")
                raw_addr = rec.get("business_address", "")
                country = rec.get("country", "")

                s1_data[eid] = {
                    "name": self.name_norm.normalize(raw_name),
                    "addr": self.addr_norm.normalize(raw_addr),
                    "comp": self.addr_comp.extract_components(raw_addr, country),
                    "country": country,
                }
                if len(s1_data) == len(s1_ids_selected):
                    break

        print(f"Normalized {len(s1_data)} Source 1 records.")

        # 3. Ingest Target records (Source 2 and Source 3) & build Blocking Index
        print("\n[Step 3/8] Ingesting Target Records & Building Multi-Pass Blocking Index...")
        generator = MultiPassCandidateGenerator()
        target_data = {}

        for source_path, prefix in [(TRAIN_SOURCE2_PATH, "S2-"), (TRAIN_SOURCE3_PATH, "S3-")]:
            count = 0
            print(f"Streaming {source_path.name}...")
            for rec in stream_tsv_records(str(source_path)):
                eid = rec["entity_id"]
                # Retain all true matches + additional pool for negatives
                is_needed = eid in target_ids_needed
                if is_needed or count < (num_target_records // 2):
                    raw_name = rec.get("business_name", "")
                    raw_addr = rec.get("business_address", "")
                    country = rec.get("country", "")

                    n_dict = self.name_norm.normalize(raw_name)
                    a_dict = self.addr_norm.normalize(raw_addr)
                    c_dict = self.addr_comp.extract_components(raw_addr, country)

                    generator.index_target_record(eid, n_dict, a_dict, c_dict)
                    target_data[eid] = {
                        "name": n_dict,
                        "addr": a_dict,
                        "comp": c_dict,
                        "country": country,
                    }
                    count += 1

        print(f"Total target records indexed: {len(target_data)}")

        # 4. Generate Candidate Pairs for all S1
        print("\n[Step 4/8] Generating Multi-Pass Candidates...")
        candidate_map: Dict[str, List[str]] = {}
        for s1_id, s1_info in s1_data.items():
            cands = generator.generate_candidates_for_query(
                s1_id=s1_id,
                norm_name=s1_info["name"],
                norm_addr=s1_info["addr"],
                addr_comp=s1_info["comp"],
            )
            candidate_map[s1_id] = cands

        # Audit blocking recall
        recall_stats = evaluate_blocking_recall(gt_map, candidate_map)
        print(f"Blocking Recall Ceiling: {recall_stats['blocking_recall'] * 100:.2f}% "
              f"({recall_stats['retained_true_matches']}/{recall_stats['total_true_matches']} true matches)")
        print(f"Average candidates per S1: {recall_stats['avg_candidates_per_entity']}")

        # 5. Entity-Aware Train / Validation Split
        print("\n[Step 5/8] Entity-Aware Data Split (80% Train / 20% Validation)...")
        all_s1_list = list(s1_data.keys())
        train_s1_ids, val_s1_ids = split_source1_entities(all_s1_list, train_ratio=0.80, seed=RANDOM_SEED)
        print(f"Train S1 Entities: {len(train_s1_ids)} | Validation S1 Entities: {len(val_s1_ids)}")

        # 6. Build Pairs & Feature Matrices
        print("\n[Step 6/8] Building Training & Validation Pairs and Extracting Features...")
        train_pairs, y_train = build_training_pairs(gt_map, candidate_map, train_s1_ids)
        val_pairs, y_val = build_training_pairs(gt_map, candidate_map, val_s1_ids)

        print(f"Constructed {len(train_pairs)} training pairs (Pos: {np.sum(y_train==1)}, Neg: {np.sum(y_train==0)})")
        print(f"Constructed {len(val_pairs)} validation pairs (Pos: {np.sum(y_val==1)}, Neg: {np.sum(y_val==0)})")

        def compute_matrix(pairs):
            X_list = []
            valid_pairs = []
            for s1_id, cid in pairs:
                s1_info = s1_data.get(s1_id)
                t_info = target_data.get(cid)
                if not s1_info or not t_info:
                    continue
                vec = extract_pairwise_feature_vector(
                    s1_name=s1_info["name"],
                    cand_name=t_info["name"],
                    s1_addr=s1_info["addr"],
                    cand_addr=t_info["addr"],
                    s1_comp=s1_info["comp"],
                    cand_comp=t_info["comp"],
                    s1_country=s1_info["country"],
                    cand_country=t_info["country"],
                    cand_id=cid,
                )
                X_list.append(vec)
                valid_pairs.append((s1_id, cid))
            return np.array(X_list, dtype=np.float32), valid_pairs

        X_train, clean_train_pairs = compute_matrix(train_pairs)
        X_val, clean_val_pairs = compute_matrix(val_pairs)

        # 7. Train XGBoost & LightGBM Models
        print("\n[Step 7/8] Training Primary XGBoost and Secondary LightGBM Models...")
        print("Training XGBoost...")
        xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)

        print("Training LightGBM...")
        lgbm_model = train_lightgbm_model(X_train, y_train, X_val, y_val)

        # Hard negative mining step
        if enable_hard_negatives:
            print("Mining Hard Negatives with XGBoost...")
            val_xgb_probs = xgb_model.predict_proba(X_train)[:, 1]
            hard_negs = mine_hard_negatives(clean_train_pairs, val_xgb_probs, gt_map, min_prob_threshold=0.30)
            if hard_negs:
                print(f"Discovered {len(hard_negs)} hard negative pairs. Augmenting training set & retraining...")
                # Re-train with mined hard negatives
                X_hard, _ = compute_matrix(hard_negs)
                y_hard = np.zeros(len(X_hard), dtype=np.int32)
                X_train_augmented = np.vstack([X_train, X_hard])
                y_train_augmented = np.concatenate([y_train, y_hard])
                xgb_model = train_xgboost_model(X_train_augmented, y_train_augmented, X_val, y_val)
                lgbm_model = train_lightgbm_model(X_train_augmented, y_train_augmented, X_val, y_val)

        # 8. Joint (alpha, threshold) Search
        print("\n[Step 8/8] Performing Joint (alpha, threshold) Search for Maximum Validation Macro F_0.5...")
        val_xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        val_lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

        val_gt_subset = {s1_id: gt_map[s1_id] for s1_id in val_s1_ids if s1_id in gt_map}
        search_res = search_optimal_alpha_threshold(
            val_pairs=clean_val_pairs,
            xgb_probs=val_xgb_probs,
            lgbm_probs=val_lgbm_probs,
            ground_truth=val_gt_subset,
            all_val_s1_ids=val_s1_ids,
        )

        best_alpha = search_res["best_alpha"]
        best_tau = search_res["best_threshold"]
        best_f05 = search_res["best_val_macro_f05"]

        print(f"\n=======================================================")
        print(f"Optimal Ensemble Weight alpha* : {best_alpha}")
        print(f"Optimal Decision Threshold tau*: {best_tau}")
        print(f"Peak Validation Macro F_0.5    : {best_f05:.5f}")
        print(f"=======================================================")

        # Build final Hybrid model
        self.hybrid_model = HybridEntityResolver(
            xgb_model=xgb_model,
            lgbm_model=lgbm_model,
            alpha=best_alpha,
            threshold=best_tau,
            feature_names=FEATURE_NAMES,
        )

        # Save model artifact
        model_save_path = self.models_dir / "hybrid_resolver.joblib"
        self.hybrid_model.save(model_save_path)
        print(f"Saved frozen hybrid model to {model_save_path}")

        # Save metrics to experiments
        metrics_report = {
            "validation_macro_f05": best_f05,
            "optimal_alpha": best_alpha,
            "optimal_threshold": best_tau,
            "blocking_recall": recall_stats["blocking_recall"],
            "avg_candidates_per_entity": recall_stats["avg_candidates_per_entity"],
            "total_train_pairs": len(train_pairs),
            "total_val_pairs": len(val_pairs),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        with open(EXPERIMENTS_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)

        return metrics_report

    def run_full_inference(
        self,
        max_test_s1: int = None,
        model_path: Path = None
    ):
        """Execute full inference over the official test set."""
        if self.hybrid_model is None:
            m_path = model_path or (self.models_dir / "hybrid_resolver.joblib")
            if not m_path.exists():
                raise FileNotFoundError(f"Model file not found at {m_path}. Train the model first.")
            print(f"Loading hybrid resolver from {m_path}...")
            self.hybrid_model = HybridEntityResolver.load(m_path)

        print("Executing test inference pipeline...")
        from inference.predict import run_test_inference
        run_test_inference(
            hybrid_model=self.hybrid_model,
            max_s1=max_test_s1,
            output_matching_path=MATCHING_RESULTS_PATH,
            output_candidate_path=CANDIDATE_PAIRS_PATH,
        )

def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all", help="Pipeline execution mode")
    parser.add_argument("--s1-samples", type=int, default=25000, help="Source 1 training sample size")
    parser.add_argument("--target-samples", type=int, default=100000, help="Target S2/S3 sample size")
    parser.add_argument("--max-test-s1", type=int, default=None, help="Max test S1 entities to process for quick run")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline()

    if args.mode in ["train", "all"]:
        pipeline.run_training_and_validation(
            num_s1_records=args.s1_samples,
            num_target_records=args.target_samples,
        )

    if args.mode in ["infer", "all"]:
        pipeline.run_full_inference(max_test_s1=args.max_test_s1)

if __name__ == "__main__":
    main()
