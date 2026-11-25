# mypy: disable-error-code=import-untyped

"""Non-overlapping forward monthly excess returns on XNYS sessions."""

from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ingen_qr.evaluation.splitters import HOLDOUT_START

YEAR_SECONDS = 365.0 * 86400.0


def horizon_labels(
    labels: pd.DataFrame, prices_path: Path, sessions: int
) -> pd.DataFrame:
    """Return the excess return of the month ending on the given session."""
    if sessions not in {21, 42, 63}:
        raise ValueError("Use 21, 42, or 63 sessions.")
    if (labels["date"].ge(HOLDOUT_START) | labels["label_end"].ge(HOLDOUT_START)).any():
        raise PermissionError("Input label windows must end before the holdout.")
    if sessions == 21:
        return labels[["date", "ticker", "label_end", "label_excess_return"]].copy()

    dates = pd.DatetimeIndex(labels["date"].unique())
    calendar = xcals.get_calendar(
        "XNYS",
        start=dates.min() - pd.Timedelta(days=10),
        end=HOLDOUT_START + pd.Timedelta(days=120),
    )
    market_days = pd.DatetimeIndex(calendar.sessions)
    positions = market_days.get_indexer(dates)
    if np.any(positions < 0):
        raise ValueError("A forecast date does not match an XNYS session.")
    window_start = market_days.take(positions + sessions - 22)
    window_end = market_days.take(positions + sessions - 1)
    entry = pd.DataFrame(
        {"date": dates, "window_start": window_start, "horizon_end": window_end}
    )
    entry = entry.loc[entry["horizon_end"] < HOLDOUT_START].copy()
    if entry.empty:
        return labels.iloc[0:0][
            ["date", "ticker", "label_end", "label_excess_return"]
        ].copy()

    entry["years"] = [
        (calendar.session_close(last) - calendar.session_close(first)).total_seconds()
        / YEAR_SECONDS
        for first, last in zip(entry["window_start"], entry["horizon_end"], strict=True)
    ]
    base_years = {
        first: (
            calendar.session_close(last) - calendar.session_open(first)
        ).total_seconds()
        / YEAR_SECONDS
        for first, last in labels[["date", "label_end"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    prices = pd.read_parquet(
        prices_path,
        columns=["date", "ticker", "adjusted_close"],
        filters=[("date", "<", HOLDOUT_START)],
    )
    if prices["date"].ge(HOLDOUT_START).any():
        raise PermissionError(
            "A price from the holdout reached the horizon calculation."
        )
    price_dates = set(entry["window_start"]) | set(entry["horizon_end"])
    prices = prices.loc[prices["date"].isin(price_dates)]
    frame = labels.merge(entry, on="date", validate="many_to_one")
    frame = frame.merge(
        prices.rename(
            columns={"date": "window_start", "adjusted_close": "start_close"}
        ),
        on=["window_start", "ticker"],
        validate="many_to_one",
    ).merge(
        prices.rename(columns={"date": "horizon_end", "adjusted_close": "end_close"}),
        on=["horizon_end", "ticker"],
        validate="many_to_one",
    )
    rate = frame["risk_free_21d"] / frame["date"].map(base_years)
    gross = frame["end_close"] / frame["start_close"]
    valid = frame["start_close"].gt(0) & frame["end_close"].gt(0) & np.isfinite(gross)
    frame["label_excess_return"] = gross - 1.0 - rate * frame["years"]
    frame["label_end"] = frame["horizon_end"]
    return frame.loc[valid, ["date", "ticker", "label_end", "label_excess_return"]]
