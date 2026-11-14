import math

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from ingen_qr.features.liquidity import liquidity_features


@pytest.fixture
def prices() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS", start="2014-01-01", end="2014-12-31")
    dates = pd.DatetimeIndex(calendar.sessions[:61]).tz_localize(None)
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "available_at": pd.to_datetime(dates, utc=True) + pd.Timedelta(hours=23),
            "adjusted_close": 100.0 * 1.01 ** np.arange(len(dates)),
            "raw_close": 10.0,
            "volume": 200.0,
        }
    )


@pytest.mark.parametrize(
    ("feature", "expected"),
    [("amihud_21d", 5.0), ("log_dollar_volume_21d", math.log(2000.0))],
)
def test_hand_calculated_liquidity(
    prices: pd.DataFrame, feature: str, expected: float
) -> None:
    assert liquidity_features(prices)[feature].iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("feature", ["amihud_21d", "log_dollar_volume_21d"])
def test_minimum_count_boundary(prices: pd.DataFrame, feature: str) -> None:
    changed = prices.copy()
    changed.loc[[42, 45, 48, 51], "volume"] = np.nan
    assert pd.notna(liquidity_features(changed)[feature].iloc[-1])
    changed.loc[54, "volume"] = np.nan
    assert pd.isna(liquidity_features(changed)[feature].iloc[-1])


def test_zero_volume_rules(prices: pd.DataFrame) -> None:
    changed = prices.copy()
    changed.loc[[42, 45, 48, 51], "volume"] = 0.0
    result = liquidity_features(changed).iloc[-1]
    assert result["amihud_21d"] == pytest.approx(5.0)
    assert result["log_dollar_volume_21d"] == pytest.approx(math.log(2000.0 * 17 / 21))
    changed.loc[54, "volume"] = 0.0
    assert pd.isna(liquidity_features(changed)["amihud_21d"].iloc[-1])
    changed["volume"] = 0.0
    result = liquidity_features(changed).iloc[-1]
    assert pd.isna(result["amihud_21d"])
    assert pd.isna(result["log_dollar_volume_21d"])


def test_missing_session_is_not_bridged(prices: pd.DataFrame) -> None:
    changed = prices.copy()
    changed["adjusted_close"] = 100.0
    changed.loc[changed.index > 45, "adjusted_close"] = 200.0
    changed = changed.drop(index=45)
    result = liquidity_features(changed)
    assert result["amihud_21d"].iloc[-1] == pytest.approx(0.0)


def test_future_data_does_not_change_past(prices: pd.DataFrame) -> None:
    cutoff = prices.loc[40, "date"]
    expected = liquidity_features(prices.loc[prices["date"] <= cutoff])
    changed = prices.copy()
    changed.loc[
        changed["date"] > cutoff, ["adjusted_close", "raw_close", "volume"]
    ] *= 10
    actual = liquidity_features(changed)
    pd.testing.assert_frame_equal(
        actual.loc[actual["date"] <= cutoff].reset_index(drop=True), expected
    )


def test_independent_tickers_and_unchanged_input(prices: pd.DataFrame) -> None:
    other = prices.assign(ticker="IBM", volume=400.0)
    both = pd.concat([prices, other], ignore_index=True).sample(frac=1, random_state=2)
    original = both.copy(deep=True)
    result = liquidity_features(both)
    for ticker, expected in [("AAPL", prices), ("IBM", other)]:
        pd.testing.assert_frame_equal(
            result.loc[result["ticker"] == ticker].reset_index(drop=True),
            liquidity_features(expected),
        )
    pd.testing.assert_frame_equal(both, original)


@pytest.mark.parametrize("feature", ["amihud_21d", "log_dollar_volume_21d"])
def test_unavailable_data_is_excluded(prices: pd.DataFrame, feature: str) -> None:
    changed = prices.copy()
    changed.loc[[42, 45, 48, 51, 54], "available_at"] = prices[
        "available_at"
    ].max() + pd.Timedelta(days=1)
    result = liquidity_features(changed)
    assert pd.isna(result[feature].iloc[-1])


def test_negative_volume_is_rejected(prices: pd.DataFrame) -> None:
    changed = prices.copy()
    changed.loc[50, "volume"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        liquidity_features(changed)
