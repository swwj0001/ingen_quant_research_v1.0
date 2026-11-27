"""Paired monthly inference with the existing Week 3 HAC convention."""

from math import erfc, sqrt
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.evaluation.signal_metrics import newey_west


def paired_monthly(left: pd.Series, right: pd.Series, lags: int = 6) -> dict[str, Any]:
    """Test left minus right on unique, finite matched calendar months."""
    series = []
    for values in (left, right):
        month = (
            values.index.asfreq("M")
            if isinstance(values.index, pd.PeriodIndex)
            else pd.PeriodIndex(pd.to_datetime(values.index), freq="M")
        )
        if month.has_duplicates or month.isna().any():
            raise ValueError(
                "Paired monthly tests require one nonmissing month key per series."
            )
        series.append(pd.Series(values.to_numpy(dtype=float), index=month))
    pairs = pd.concat(series, axis=1, join="inner").sort_index()
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    deltas = pairs.iloc[:, 0] - pairs.iloc[:, 1]
    t = newey_west(deltas, lags)
    return {
        "paired_months": len(deltas),
        "mean_difference": float(deltas.mean()),
        "nw_t": t,
        "p_two_sided_asymptotic": erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan,
    }
