# mypy: disable-error-code=import-untyped

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from ingen_qr.features.momentum import FEATURES, momentum_features


@pytest.fixture
def prices() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS", start="2013-01-01", end="2015-12-31")
    dates = pd.DatetimeIndex(calendar.sessions[:270]).tz_localize(None)
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "adjusted_close": 100.0 + np.arange(len(dates)),
            "available_at": pd.to_datetime(dates, utc=True) + pd.Timedelta(hours=23),
        }
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("reversal_21d", -(352.0 / 331.0 - 1.0)),
        ("momentum_63_21d", 331.0 / 289.0 - 1.0),
        ("momentum_126_21d", 331.0 / 226.0 - 1.0),
        ("momentum_252_21d", 331.0 / 100.0 - 1.0),
    ],
)
def test_hand_calculated_values(
    prices: pd.DataFrame, name: str, expected: float
) -> None:
    result = momentum_features(prices)
    assert result.loc[252, name] == pytest.approx(expected)


def test_warmup_is_missing(prices: pd.DataFrame) -> None:
    result = momentum_features(prices)
    for name, (_, lookback, _) in FEATURES.items():
        assert result.loc[: lookback - 1, name].isna().all()
        assert pd.notna(result.loc[lookback, name])


def test_tickers_are_independent_and_input_is_unchanged(prices: pd.DataFrame) -> None:
    other = prices.assign(ticker="MSFT", adjusted_close=200.0 + np.arange(270) * 2.0)
    both = pd.concat([prices, other], ignore_index=True).sample(frac=1, random_state=1)
    original = both.copy(deep=True)
    result = momentum_features(both)
    pd.testing.assert_frame_equal(both, original)
    for ticker, expected in [("AAPL", prices), ("MSFT", other)]:
        actual = result.loc[result["ticker"].eq(ticker)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, momentum_features(expected))


def test_missing_session_does_not_shorten_horizon(prices: pd.DataFrame) -> None:
    date = prices.loc[252, "date"]
    missing_inside_window = momentum_features(prices.drop(index=230)).set_index("date")
    assert missing_inside_window.loc[date, "momentum_252_21d"] == pytest.approx(
        331.0 / 100.0 - 1.0
    )
    missing_endpoint = momentum_features(prices.drop(index=231)).set_index("date")
    assert missing_endpoint.loc[date, list(FEATURES)].isna().all()


def test_future_prices_cannot_change_past_features(prices: pd.DataFrame) -> None:
    cutoff = prices.loc[200, "date"]
    expected = momentum_features(prices.loc[prices["date"] <= cutoff])
    changed = prices.copy()
    changed.loc[changed["date"] > cutoff, "adjusted_close"] *= 50.0
    actual = momentum_features(changed)
    actual = actual.loc[actual["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_unavailable_historical_endpoint_is_not_used(prices: pd.DataFrame) -> None:
    delayed = prices.copy()
    delayed.loc[231, "available_at"] = prices.loc[252, "available_at"] + pd.Timedelta(
        days=1
    )
    result = momentum_features(delayed)
    assert result.loc[252, list(FEATURES)].isna().all()


def test_duplicate_records_require_asof_selection(prices: pd.DataFrame) -> None:
    duplicate = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="as-of"):
        momentum_features(duplicate)


@pytest.mark.parametrize("timestamp_type", ["TIMESTAMPTZ", "TIMESTAMPTZ_NS"])
def test_readonly_database_smoke_check(
    prices: pd.DataFrame, tmp_path: Path, timestamp_type: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "data").mkdir()
    future = prices.iloc[[-1]].assign(
        date=pd.Timestamp("2025-01-02"),
        available_at=pd.Timestamp("2025-01-02T23:00:00Z"),
        adjusted_close=999999.0,
    )
    panel = pd.concat([prices, future], ignore_index=True)
    with duckdb.connect(str(tmp_path / "data/ingen_qr.duckdb")) as connection:
        connection.register("panel", panel)
        connection.execute(
            "CREATE TABLE prices AS SELECT ticker, date, adjusted_close, "
            f"CAST(available_at AS {timestamp_type}) AS available_at FROM panel"
        )
        late = (
            "2014-12-31T23:59:59.000000001Z"
            if timestamp_type == "TIMESTAMPTZ_NS"
            else "2015-01-01T00:00:00Z"
        )
        connection.execute(
            "INSERT INTO prices VALUES ('AAPL', DATE '2014-12-31', 999999.0, "
            f"CAST(? AS {timestamp_type}))",
            [late],
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    checked = subprocess.run(
        [sys.executable, str(project / "scripts/check_momentum.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Rows: 270" in checked.stdout
    assert "momentum_252_21d" in checked.stdout
    assert "2025-01-02" not in checked.stdout
