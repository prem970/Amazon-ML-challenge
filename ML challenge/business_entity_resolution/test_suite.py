"""
Unit and integration test suite for the Business Entity Resolution project.
Validates normalization, blocking, feature extraction, model fitting, and metrics.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    normalize_unicode,
    clean_text,
    NameNormalizer,
    AddressNormalizer,
    AddressComponentExtractor,
)
from blocking import (
    ExactNameBlocker,
    TokenNameBlocker,
    AddressKeysBlocker,
    MultiPassCandidateGenerator,
)
from features import (
    levenshtein_ratio,
    jaro_winkler_sim,
    compute_name_features,
    compute_address_features,
    compute_interaction_features,
    extract_pairwise_feature_vector,
    FEATURE_NAMES,
)
from evaluation import compute_entity_f05, compute_macro_f05
from training import HybridEntityResolver

def test_preprocessing():
    print("Testing Preprocessing & Normalization...")
    # Unicode & French accents
    assert normalize_unicode("Bordeaux, Nouvelle-Aquitaine") == "bordeaux, nouvelle-aquitaine"
    assert normalize_unicode("Hôtel & Café") == "hotel & cafe"

    # Name normalization
    norm = NameNormalizer()
    res = norm.normalize("Acme Solutions Private Limited")
    assert "solutions" in res["basic"]
    assert res["legal"] == "acme"
    assert res["compact"] == "acme"

    # Address normalization
    anorm = AddressNormalizer()
    ares = anorm.normalize("123 Main St, Suite 400, Rd 5")
    assert "street" in ares["basic"]
    assert "road" in ares["basic"]

    # Component extraction
    comp = AddressComponentExtractor()
    cres = comp.extract_components("1795 Westchester Drive, High Point, NC 27262", country="US")
    assert cres["postal_code"] == "27262"
    assert cres["house_number"] == "1795"
    assert cres["state"] == "north carolina"
    print("  [PASS] Preprocessing tests passed.")

def test_blocking():
    print("Testing Multi-Pass Blocking...")
    gen = MultiPassCandidateGenerator(max_candidates=10)
    nn = NameNormalizer()
    an = AddressNormalizer()
    ac = AddressComponentExtractor()

    # Index target S2
    n2 = nn.normalize("Apex Industrial Corp")
    a2 = an.normalize("500 Market St, Denver, CO 80202")
    c2 = ac.extract_components("500 Market St, Denver, CO 80202", "US")
    gen.index_target_record("S2-001", n2, a2, c2)

    # Query S1
    n1 = nn.normalize("Apex Industrial Corporation")
    a1 = an.normalize("500 Market Street, Denver, Colorado 80202")
    c1 = ac.extract_components("500 Market Street, Denver, Colorado 80202", "US")
    cands = gen.generate_candidates_for_query("S1-001", n1, a1, c1)

    assert "S2-001" in cands
    print("  [PASS] Blocking tests passed.")

def test_feature_engineering():
    print("Testing Pairwise Feature Engineering...")
    nn = NameNormalizer()
    an = AddressNormalizer()
    ac = AddressComponentExtractor()

    n1 = nn.normalize("Apex Industrial Corporation")
    n2 = nn.normalize("Apex Industrial Corp")
    a1 = an.normalize("500 Market Street, Denver, CO")
    a2 = an.normalize("500 Market St, Denver, Colorado")
    c1 = ac.extract_components("500 Market Street, Denver, CO", "US")
    c2 = ac.extract_components("500 Market St, Denver, Colorado", "US")

    vec = extract_pairwise_feature_vector(
        s1_name=n1, cand_name=n2,
        s1_addr=a1, cand_addr=a2,
        s1_comp=c1, cand_comp=c2,
        s1_country="US", cand_country="US",
        cand_id="S2-001"
    )

    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(FEATURE_NAMES)
    assert not np.isnan(vec).any()
    print("  [PASS] Feature extraction tests passed.")

def test_metrics():
    print("Testing Official Macro F_0.5 Metric...")
    # Singleton correct
    p, r, f = compute_entity_f05(set(), set())
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # Singleton false positive
    p, r, f = compute_entity_f05(set(), {"S2-001"})
    assert (p, r, f) == (0.0, 1.0, 0.0)

    # Perfect match
    p, r, f = compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002"})
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # Partial match
    # true: {S2-00047, S3-00812}, pred: {S2-00047, S2-00193, S3-00812}
    # Precision = 2/3, Recall = 1.0 -> F0.5 = 0.714285...
    p, r, f = compute_entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(p - 2/3) < 1e-4
    assert abs(r - 1.0) < 1e-4
    assert abs(f - 0.714285) < 1e-4
    print("  [PASS] Metric tests passed.")

def test_hybrid_resolver():
    print("Testing Hybrid Resolver Ensembler...")
    class MockModel:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.85)])

    hybrid = HybridEntityResolver(
        xgb_model=MockModel(),
        lgbm_model=MockModel(),
        alpha=0.6,
        threshold=0.75,
        feature_names=FEATURE_NAMES
    )
    dummy_X = np.zeros((2, len(FEATURE_NAMES)))
    probs = hybrid.predict_proba(dummy_X)
    preds = hybrid.predict(dummy_X)
    assert len(probs) == 2
    assert np.isclose(probs[0], 0.85)
    assert (preds == [1, 1]).all()
    print("  [PASS] Hybrid Resolver tests passed.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BUSINESS ENTITY RESOLUTION TEST SUITE")
    print("=" * 60)
    test_preprocessing()
    test_blocking()
    test_feature_engineering()
    test_metrics()
    test_hybrid_resolver()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY! [5/5]")
