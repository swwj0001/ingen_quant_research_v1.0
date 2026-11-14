"""Untuned OLS-3; the pure momentum benchmark uses the same scoring cohort."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

OLS_FEATURES = ["log_market_cap", "book_to_market", "momentum_252_21d"]


def ols_predict(
    train: pd.DataFrame, target: pd.Series, test: pd.DataFrame
) -> NDArray[np.float64]:
    x = train[OLS_FEATURES].to_numpy(dtype=float)
    z = test[OLS_FEATURES].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("OLS inputs must be finite after per-date preprocessing.")
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), y, rcond=None
    )[0]
    return np.asarray(
        np.column_stack([np.ones(len(z)), z]) @ coefficients, dtype=np.float64
    )
