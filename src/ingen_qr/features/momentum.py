# mypy: disable-error-code=import-untyped

"""Trailing return features on the XNYS session grid."""

import exchange_calendars as xcals
import numpy as np
import pandas as pd

FEATURES = {
    "reversal_21d": (0, 21, -1.0),
    "momentum_63_21d": (21, 63, 1.0),
    "momentum_126_21d": (21, 126, 1.0),
    "momentum_252_21d": (21, 252, 1.0),
}
INPUTS = ["ticker", "date", "adjusted_close", "available_at"]
OUTPUTS = ["ticker", "date", "available_at", *FEATURES]


def momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Use historical endpoints known by each row's available_at timestamp.

    A month is approximated by 21 XNYS sessions. Features are available after
    the source row's close-plus-30-minute cutoff. The returned rows do not
    yet apply historical universe membership or cross-sectional preprocessing.
    """
    missing = set(INPUTS) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("No prices supplied.")

    frame = prices[INPUTS].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].dt.tz is not None:
        raise ValueError("date must contain timezone-naive trading dates.")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("date must contain midnight trading dates.")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    if frame[["ticker", "date", "available_at"]].isna().any().any():
        raise ValueError("Missing ticker, date or available_at.")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Select one as-of price record per ticker and date.")
    valid_prices = frame["adjusted_close"].dropna()
    if not np.isfinite(valid_prices).all() or (valid_prices <= 0).any():
        raise ValueError("Observed adjusted_close values must be positive and finite.")
    frame = frame.sort_values(["ticker", "date"])

    calendar = xcals.get_calendar(
        "XNYS", start=frame["date"].min(), end=frame["date"].max()
    )
    sessions = pd.DatetimeIndex(calendar.sessions).tz_localize(None)
    if not frame["date"].isin(sessions).all():
        raise ValueError("A price date is not an XNYS trading session.")

    results: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", sort=True):
        observed = pd.DatetimeIndex(group["date"])
        grid = sessions[(sessions >= observed.min()) & (sessions <= observed.max())]
        block = group.set_index("date").reindex(grid)
        price = block["adjusted_close"]
        known_at = block["available_at"]

        for name, (skip, lookback, direction) in FEATURES.items():
            first_known = known_at.shift(skip) <= known_at
            last_known = known_at.shift(lookback) <= known_at
            value = direction * (price.shift(skip) / price.shift(lookback) - 1.0)
            block[name] = value.where(first_known & last_known)

        result = block.loc[observed, ["available_at", *FEATURES]].copy()
        result["ticker"] = ticker
        result.index.name = "date"
        results.append(result.reset_index()[OUTPUTS])

    return pd.concat(results, ignore_index=True)
