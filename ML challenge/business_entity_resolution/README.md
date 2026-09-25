# Amazon ML Challenge 2026: Business Entity Resolution
## Precision-First XGBoost + LightGBM Hybrid Architecture

This repository contains the end-to-end, production-grade business entity resolution system built according to the **Complete End-to-End Architecture & High-Accuracy Implementation Plan**.

---

### Key System Highlights
- **Precision-First Design**: Tailored to optimize the official macro $F_{0.5}$ metric (which penalizes false merges $2\times$ more than missed matches).
- **Multi-View Preprocessing**: Preserves original signal across `raw`, `basic`, `legal-normalized`, `token`, `compact`, and `char_ngram` representations.
- **Multi-Pass Blocking (B1 to B9)**: Achieves high recall ceiling by unioning exact legal keys, compact name prefixes, distinctive high-information tokens, postal codes, and structured address components.
- **Rich Pairwise Features**: 48 complementary features covering string similarities (Levenshtein, Jaro-Winkler, Token Sort/Set, Jaccard, N-grams), structured address matching (postal code, house number, state), open-set country indicators, and cross-field interactions.
- **Hybrid XGBoost + LightGBM Ensemble**: Blends predictions $P = \alpha \cdot P_{XGB} + (1 - \alpha) \cdot P_{LGBM}$ and performs a joint $(\alpha, \tau)$ grid search directly on held-out validation macro $F_{0.5}$.
- **Precision Safety Layer & Singleton Abstention**: Explicitly detects singletons (worth 1.0 each) and rejects weak or contradictory candidates.
- **Zero External Lookups**: Strictly complies with competition rules without external APIs or web enrichment.

---

### Repository Structure
```
business_entity_resolution/
├── data/
│   ├── train/               # Source TSV data (train_source1, 2, 3, ground_truth)
│   └── test/                # Test TSV data (test_source1, 2, 3)
├── src/
│   ├── config.py            # Global paths, hyperparameters, seeds
│   ├── pipeline.py          # Unified end-to-end orchestration pipeline
│   ├── data/
│   │   ├── loader.py        # Memory-efficient streaming TSV reader
│   │   ├── validator.py     # Schema and ID prefix validation
│   │   └── profiling.py     # Missingness and noise profiling
│   ├── preprocessing/
│   │   ├── unicode_utils.py # Unicode NFKD & diacritic stripping (US, India, France)
│   │   ├── name_normalizer.py # Multi-view corporate legal & domain normalization
│   │   ├── address_normalizer.py # Road/street abbreviation & punctuation cleaning
│   │   └── address_components.py # Postal/PIN/ZIP, house number, state extraction
│   ├── blocking/
│   │   ├── exact_name.py    # B1 & B2 exact and prefix name blocker
│   │   ├── token_name.py    # B3 distinctive token blocker
│   │   ├── address_keys.py  # B4, B5, B6 postal, state, house number keys
│   │   ├── tfidf_retrieval.py # B7, B8, B9 soft lexical & n-gram retriever
│   │   └── candidate_generator.py # Deduplicated union & recall auditing
│   ├── features/
│   │   ├── string_similarity.py # Rapidfuzz C++ similarity primitives
│   │   ├── name_features.py # Pairwise name similarity features
│   │   ├── address_features.py # Pairwise address & component features
│   │   ├── interaction_features.py # Cross-field terms, country & source flags
│   │   └── build_features.py # 48-feature unified vector assembler
│   ├── training/
│   │   ├── split.py         # Entity-aware 80/20 train/validation split
│   │   ├── build_pairs.py   # Balanced positive & hard negative pair constructor
│   │   ├── hard_negative_mining.py # Active mining of high-probability false positives
│   │   ├── train_xgb.py     # Primary XGBoost model trainer
│   │   ├── train_lgbm.py    # Secondary LightGBM model trainer
│   │   └── hybrid_model.py  # Blended probability ensemble & persistence
│   ├── evaluation/
│   │   ├── metrics.py       # Exact Macro F_0.5 evaluator
│   │   ├── threshold_search.py # Joint (alpha, tau) grid search
│   │   ├── blocking_recall.py # Candidate recall ceiling auditor
│   │   └── error_analysis.py # False positive & false negative profiler
│   └── inference/
│       ├── safety_layer.py  # Precision guards & singleton abstention logic
│       ├── output_writer.py # TSV generator & official validator runner
│       └── predict.py       # Test set streaming inference
├── models/                  # Saved frozen model checkpoints
├── output/
│   ├── matching_results.tsv # Final entity matches (scored on leaderboard)
│   └── candidate_pairs.tsv  # Final candidates scored by the model
├── experiments/             # Metrics, reports, and logs
├── requirements.txt         # Pinned python dependencies
├── README.md                # Reproducibility instructions
└── Documentation_template.md # Official methodology document
```

---

### Step-by-Step Reproduction Guide

#### 1. Setup Environment
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

#### 2. Run End-to-End Training, Evaluation, and Inference
To train the models on training records, perform entity-aware validation, jointly optimize $(\alpha, \tau)$ for maximum Macro $F_{0.5}$, and run inference:
```bash
python src/pipeline.py --mode all
```

#### 3. Run Inference Only (Using Saved Models)
If models are already trained and saved in `models/hybrid_resolver.joblib`:
```bash
python src/pipeline.py --mode infer
```

#### 4. Validate Submission
Run the challenge submission validator directly:
```bash
python ../dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../dataset/student_resource/dataset/test
```

---

### Official Submission Packaging
The required archive can be bundled as:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
