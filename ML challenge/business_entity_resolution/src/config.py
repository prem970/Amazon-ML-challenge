"""
Configuration module for Business Entity Resolution (Amazon ML Challenge 2026).
Adheres strictly to the End-to-End Architecture specification.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "dataset" / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Output submission paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Reproducibility
RANDOM_SEED = 42

# Challenge Constraints & Constants
DELIMITER = "\t"
S1_PREFIX = "S1-"
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"
VALID_SOURCES = {S1_PREFIX, S2_PREFIX, S3_PREFIX}
KNOWN_TRAIN_COUNTRIES = {"US", "India"}

# Blocking / Candidate Generation Settings
BLOCKING_CONFIG = {
    "max_candidates_per_entity": 25,
    "min_name_prefix_len": 6,
    "max_block_bucket_size": 200,   # Prevent oversized candidate explosion on common words
    "tfidf_max_features": 15000,
    "tfidf_top_k": 5,
    "use_multiprocessing": True,
}

# XGBoost Hyperparameters (Architecture Doc Section 11)
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "gamma": 0.10,
    "reg_alpha": 0.10,
    "reg_lambda": 2.0,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM Hyperparameters (Architecture Doc Section 12)
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 1.0,
    "objective": "binary",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Hybrid Search Settings (Section 13)
HYBRID_CONFIG = {
    "alpha_coarse": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
    "threshold_coarse_min": 0.50,
    "threshold_coarse_max": 0.96,
    "threshold_coarse_step": 0.02,
    "threshold_fine_step": 0.005,
    "default_alpha": 0.55,
    "default_threshold": 0.72,
}

# Precision Safety Settings (Section 16 & 17)
SAFETY_CONFIG = {
    "contradictory_country_penalty": True,
    "require_strong_address_if_name_weak": True,
    "min_name_similarity_for_pure_name_match": 0.88,
    "min_addr_similarity_for_pure_addr_match": 0.90,
}
