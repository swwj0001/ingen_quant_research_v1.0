from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ingen_qr.evaluation.week3_horizons import YEAR_SECONDS, horizon_labels
from ingen_qr.evaluation.week3_industry import classify_sic
from ingen_qr.evaluation.week3_metrics import classification, extended_monthly


def test_long_horizon_is_censored_before_reading_holdout_prices(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-11-01"]),
            "ticker": ["AAA"],
            "label_end": pd.to_datetime(["2023-11-30"]),
            "label_excess_return": [0.01],
        }
    )
    assert horizon_labels(labels, tmp_path / "not_opened.parquet", 63).empty
    assert len(horizon_labels(labels, tmp_path / "not_opened.parquet", 21)) == 1


def test_decay_uses_disjoint_close_to_close_windows(tmp_path: Path) -> None:
    calendar = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2023-07-03")
    session = calendar.sessions.get_loc(start)
    day_21, day_42, day_63 = calendar.sessions.take(
        [session + 20, session + 41, session + 62]
    )
    rate = 0.06
    base_years = (
        calendar.session_close(day_21) - calendar.session_open(start)
    ).total_seconds() / YEAR_SECONDS
    labels = pd.DataFrame(
        {
            "date": [start],
            "ticker": ["AAA"],
            "label_end": [day_21],
            "label_excess_return": [0.28],
            "risk_free_21d": [rate * base_years],
        }
    )
    prices = tmp_path / "prices.parquet"
    pd.DataFrame(
        {
            "date": [start, day_21, day_42, day_63],
            "ticker": ["AAA"] * 4,
            "adjusted_close": [100.0, 120.0, 132.0, 99.0],
        }
    ).to_parquet(prices, index=False)

    first = horizon_labels(labels, prices, 21).iloc[0]
    assert first["label_excess_return"] == 0.28
    for sessions, start_close, end_close, gross in (
        (42, day_21, day_42, 132 / 120),
        (63, day_42, day_63, 99 / 132),
    ):
        row = horizon_labels(labels, prices, sessions).iloc[0]
        years = (
            calendar.session_close(end_close) - calendar.session_close(start_close)
        ).total_seconds() / YEAR_SECONDS
        assert row["label_end"] == end_close
        assert np.isclose(row["label_excess_return"], gross - 1 - rate * years)


def test_ff12_and_missing_sic_follow_separate_coverage_rules() -> None:
    bins = pd.read_csv("configs/week3_ff12_sic.csv")
    assert classify_sic("3674", bins) == (3674, "BusEq")
    assert classify_sic("6020", bins) == (6020, "Money")
    assert classify_sic("2834", bins) == (2834, "Hlth")
    assert classify_sic("", bins) == (None, "Other")
    assert classify_sic("1500", bins) == (1500, "Other")

    scores = np.arange(30, dtype=float)
    frame = pd.DataFrame(
        {
            "model": "ridge",
            "date": pd.Timestamp("2023-01-03"),
            "ticker": [f"T{i}" for i in range(30)],
            "prediction": scores,
            "label_excess_return": scores,
            "industry": ["Money"] * 15 + ["Other"] * 15,
            "sic_available": [True] * 15 + [False] * 15,
            "label_class": [0.0] * 10 + [np.nan] * 10 + [1.0] * 10,
        }
    )
    monthly, _ = extended_monthly(frame)
    assert np.isclose(monthly.loc[0, "industry_neutral_ic"], 1.0)
    assert monthly.loc[0, "sic_coverage"] == 0.5
    metrics = classification(frame)
    assert metrics.loc[0, "classification_coverage"] == 1
    assert metrics.loc[0, "auc"] == 1
