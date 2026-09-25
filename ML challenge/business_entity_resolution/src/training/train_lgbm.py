"""
LightGBM model training module.
Secondary tree model with complementary leaf-wise growth for hybrid ensembling.
"""

from typing import Optional
import numpy as np
import lightgbm as lgb
try:
    from ..config import LGBM_PARAMS
except (ImportError, ValueError):
    from config import LGBM_PARAMS

def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    custom_params: Optional[dict] = None,
) -> lgb.LGBMClassifier:
    """Train LightGBM binary classifier with early stopping."""
    params = LGBM_PARAMS.copy()
    if custom_params:
        params.update(custom_params)

    model = lgb.LGBMClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model
