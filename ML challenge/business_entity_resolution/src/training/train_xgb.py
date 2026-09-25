"""
XGBoost model training module.
Primary gradient boosted classifier over pairwise engineered features.
"""

from typing import Optional, Tuple
import numpy as np
import xgboost as xgb
try:
    from ..config import XGB_PARAMS
except (ImportError, ValueError):
    from config import XGB_PARAMS

def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    custom_params: Optional[dict] = None,
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier with early stopping and logloss optimization."""
    params = XGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)

    # Calculate scale_pos_weight if needed
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    if pos_count > 0 and neg_count > 0:
        params["scale_pos_weight"] = 1.0  # Precision-first: avoid artificially over-weighting positives

    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model
