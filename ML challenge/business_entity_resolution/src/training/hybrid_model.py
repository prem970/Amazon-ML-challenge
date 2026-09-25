"""
Hybrid Ensemble Model combining XGBoost and LightGBM.
Manages blended probability calculation, persistence, and inference scoring.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np

class HybridEntityResolver:
    def __init__(
        self,
        xgb_model=None,
        lgbm_model=None,
        alpha: float = 0.5,
        threshold: float = 0.70,
        feature_names: Optional[list] = None,
    ):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.feature_names = feature_names or []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute hybrid blended probability P = alpha * P_XGB + (1 - alpha) * P_LGBM."""
        if self.xgb_model is not None and self.lgbm_model is not None:
            p_xgb = self.xgb_model.predict_proba(X)[:, 1]
            p_lgb = self.lgbm_model.predict_proba(X)[:, 1]
            return self.alpha * p_xgb + (1.0 - self.alpha) * p_lgb
        elif self.xgb_model is not None:
            return self.xgb_model.predict_proba(X)[:, 1]
        elif self.lgbm_model is not None:
            return self.lgbm_model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No underlying model has been trained or loaded.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary match decisions based on optimal threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def save(self, filepath: Path):
        """Save model bundle to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "xgb_model": self.xgb_model,
            "lgbm_model": self.lgbm_model,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "HybridEntityResolver":
        """Load model bundle from disk."""
        bundle = joblib.load(filepath)
        return cls(
            xgb_model=bundle.get("xgb_model"),
            lgbm_model=bundle.get("lgbm_model"),
            alpha=bundle.get("alpha", 0.5),
            threshold=bundle.get("threshold", 0.70),
            feature_names=bundle.get("feature_names", []),
        )
