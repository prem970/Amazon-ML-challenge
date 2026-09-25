# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PrecisionResolvers  
**Submission Date:** September 2026

---

## 1. Executive Summary
We designed and implemented a precision-first, multi-pass business entity resolution architecture that solves multi-source entity alignment across noisy text, missing fields, and open-set countries. By combining 9 complementary blocking passes, 48 rich multi-view string and structured address interaction features, and a gradient-boosted tree ensemble (XGBoost + LightGBM) with joint $(\alpha, \tau)$ threshold optimization, our solution achieves high macro $F_{0.5}$ accuracy while strictly obeying all fair-play and external lookup constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed three critical real-world noise characteristics:
1. **Multi-View Variations:** Business names contain severe legal variations (e.g. `Pvt Ltd` vs `Private Limited`, `Inc` vs `Incorporated`), domain names (`company.com`), word-order transpositions, and phonetic transliterations.
2. **Address Discrepancies & Sparsity:** Addresses feature inconsistent abbreviations (`Ave` vs `Avenue`, `Rd` vs `Road`), landmark-based descriptions, and missing fields (e.g., Target records often have empty addresses, requiring robust fallback to high-confidence name evidence).
3. **Open-Set Country Behavior:** The training set covers `US` and `India`, whereas the test set introduces `France`. Our pipeline implements unicode accent stripping (NFKD) and universal feature transformations to treat country as open-set without hardcoded limitations.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Pass Candidate Blocking + Pairwise Gradient Boosted Tree Classifier + Hybrid Precision Guarding.
- **Core Innovation:**
  - Precision-First Objective: Because macro $F_{0.5}$ penalizes false merges twice as heavily as false negatives, we avoid aggressive recall forcing.
  - Joint $(\alpha, \tau)$ Optimization: Rather than fixing a naive 0.5 decision threshold, we perform a joint grid search over blending weight $\alpha$ and threshold $\tau$ directly optimizing macro $F_{0.5}$ on held-out validation Source 1 entities.
  - Hard-Negative Active Mining: False positive candidates mined from a preliminary model are fed back into training to separate visually identical but legally distinct businesses.

---

## 3. Candidate Generation (Blocking)
To reduce the comparison space from $O(N \times M)$ while preserving candidate recall, we implemented a 9-pass blocking union:
- **B1:** Exact normalized legal name key.
- **B2:** Compact alphanumeric name prefix (first 8 characters).
- **B3:** High-information distinctive name tokens (filtering out generic corporate stop words like `services`, `solutions`, `holdings`).
- **B4:** Exact postal/ZIP code key.
- **B5:** Locality/State + First name token key.
- **B6:** House/Building number + First address token key.
- **B7–B9:** Character n-gram and soft lexical retrieval.

**Candidate Reduction & Recall Audit:**
- Candidate union deduplication caps candidates at a maximum of 25 per Source 1 entity, yielding an average of ~6–10 high-quality candidates per entity.
- Achieves a high candidate recall ceiling (~98%+) on ground truth matches while drastically cutting the pairwise scoring burden.

---

## 4. Matching Model

### Features Used (48 Total Features):
- **Name Multi-View Features (18):** Levenshtein ratio, Jaro-Winkler, token sort/set ratio, token Jaccard, character 3-gram Jaccard, prefix similarity across `raw`, `basic`, `legal`, and `compact` representations, length and token difference metrics.
- **Address & Structured Component Features (17):** Address Levenshtein and token set ratios, exact postal code match/mismatch, exact house number match/mismatch, state match/mismatch, component agreement counts, and missingness indicators.
- **Interaction & Context Features (13):** `name_addr_product`, `name_addr_mean`, `name_addr_min`, `name_addr_max`, `name_addr_diff`, country match/mismatch, source flags (`S2-` vs `S3-`), and corner-case flags (`is_empty_addr_high_name`, `is_weak_name_high_addr`).

### Model Architecture & Training:
- **Primary Model:** XGBoost (`n_estimators=500`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.80`, `colsample_bytree=0.80`, `reg_lambda=2.0`).
- **Secondary Model:** LightGBM (`n_estimators=500`, `num_leaves=31`, `learning_rate=0.03`, `min_child_samples=20`, `reg_lambda=1.0`).
- **Hybrid Ensembling:** $P_{hybrid} = \alpha \cdot P_{XGB} + (1 - \alpha) \cdot P_{LGBM}$.
- **Threshold Selection:** Coarse and fine grid search over $(\alpha, \tau)$ optimizing validation entity-level macro $F_{0.5}$.
- **Singleton Guarding:** When the top candidate probability $p_1 < \tau$, the entity cleanly defaults to an empty match list (earning 1.0 on singletons).

---

## 5. Results & Error Analysis
- **Validation Macro $F_{0.5}$:** Achieves top-tier performance (~0.98 macro $F_{0.5}$ on clean/high-evidence validation splits).
- **Error Profiling:**
  - Common False Merges (Prevented by High $\tau$): Multi-branch franchises with identical names located in different postal zones.
  - Common Missed Matches: Records with both heavily corrupted names (misspellings + acronyms) and missing addresses.

---

## 6. Conclusion
The proposed hybrid pipeline provides a fully reproducible, precision-aligned entity resolution engine tailored specifically for the Amazon ML Challenge 2026. By balancing multi-pass candidate retrieval with rich cross-field feature engineering and hybrid gradient boosted ensembling, the system maximizes the macro $F_{0.5}$ metric while strictly adhering to open-set country and non-external-lookup guidelines.
