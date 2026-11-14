import math

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from ingen_qr.features.risk import FEATURES, risk_features


@pytest.fixture
def panel() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS", start="2013-01-01", end="2015-12-31")
    dates = pd.DatetimeIndex(calendar.sessions[:271]).tz_localize(None)
    x = np.resize(np.array([-0.02, 0.0, 0.02]), 270)
    noise = np.resize(np.array([0.01, -0.02, 0.01]), 270)
    y = 0.001 + 1.5 * x + noise
    blocks = []
    for ticker, returns in [("AAPL", y), ("SPY", x)]:
        blocks.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "available_at": pd.to_datetime(dates, utc=True)
                    + pd.Timedelta(hours=23),
                    "adjusted_close": 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns]),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("volatility_21d", math.sqrt(0.0008 * 21 / 20)),
        ("volatility_60d", math.sqrt(0.0008 * 60 / 59)),
        ("beta_252d", 1.5),
        ("idio_volatility_252d", math.sqrt(0.0002 * 252 / 251)),
        ("max_return_21d", 0.041),
    ],
)
def test_hand_calculated_risk(
    panel: pd.DataFrame, feature: str, expected: float
) -> None:
    result = risk_features(panel)
    value = result.loc[result["ticker"] == "AAPL", feature].iloc[-1]
    assert value == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    ("feature", "width", "holes"),
    [
        ("volatility_21d", 21, 2),
        ("max_return_21d", 21, 2),
        ("volatility_60d", 60, 6),
        ("beta_252d", 252, 25),
        ("idio_volatility_252d", 252, 25),
    ],
)
def test_minimum_count_boundary(
    panel: pd.DataFrame, feature: str, width: int, holes: int
) -> None:
    dates = panel.loc[panel["ticker"] == "AAPL", "date"].reset_index(drop=True)
    start = len(dates) - width
    missing = dates.iloc[[start + 2 + 3 * i for i in range(holes)]]
    changed = panel.copy()
    changed.loc[
        changed["ticker"].eq("AAPL") & changed["date"].isin(missing), "adjusted_close"
    ] = np.nan
    enough = risk_features(changed)
    assert pd.notna(enough.loc[enough["ticker"] == "AAPL", feature].iloc[-1])
    changed.loc[
        changed["ticker"].eq("AAPL") & changed["date"].eq(dates.iloc[start - 1]),
        "adjusted_close",
    ] = np.nan
    insufficient = risk_features(changed)
    assert pd.isna(insufficient.loc[insufficient["ticker"] == "AAPL", feature].iloc[-1])


def test_missing_session_is_not_bridged(panel: pd.DataFrame) -> None:
    dates = panel.loc[panel["ticker"] == "AAPL", "date"].reset_index(drop=True)
    changed = panel.copy()
    stock = changed["ticker"].eq("AAPL")
    changed.loc[stock, "adjusted_close"] = 100.0
    changed.loc[stock & (changed["date"] > dates.iloc[180]), "adjusted_close"] = 200.0
    changed = changed.loc[~(stock & changed["date"].eq(dates.iloc[180]))]
    result = risk_features(changed)
    row = result.loc[result["ticker"].eq("AAPL") & result["date"].eq(dates.iloc[190])]
    assert row["max_return_21d"].item() == pytest.approx(0.0)
    assert row["volatility_21d"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("ticker", ["AAPL", "SPY"])
def test_future_prices_do_not_change_past(panel: pd.DataFrame, ticker: str) -> None:
    cutoff = panel.loc[panel["ticker"] == "AAPL", "date"].iloc[230]
    expected = risk_features(panel.loc[panel["date"] <= cutoff])
    changed = panel.copy()
    changed.loc[
        changed["ticker"].eq(ticker) & (changed["date"] > cutoff), "adjusted_close"
    ] *= 50.0
    actual = risk_features(changed)
    pd.testing.assert_frame_equal(
        actual.loc[actual["date"] <= cutoff].reset_index(drop=True), expected
    )


def test_independent_tickers_and_unchanged_input(panel: pd.DataFrame) -> None:
    other = panel.loc[panel["ticker"] == "AAPL"].assign(ticker="IBM")
    other["adjusted_close"] = np.linspace(100.0, 300.0, len(other))
    changed = pd.concat([panel, other], ignore_index=True).sample(
        frac=1, random_state=2
    )
    original = changed.copy(deep=True)
    actual = risk_features(changed)
    expected = risk_features(panel)
    pd.testing.assert_frame_equal(
        actual.loc[actual["ticker"] != "IBM"].reset_index(drop=True), expected
    )
    pd.testing.assert_frame_equal(changed, original)


def test_unavailable_spy_prices_are_excluded(panel: pd.DataFrame) -> None:
    dates = panel.loc[panel["ticker"] == "SPY", "date"].reset_index(drop=True)
    changed = panel.copy()
    mask = changed["ticker"].eq("SPY") & changed["date"].isin(dates.iloc[30:108:3])
    changed.loc[mask, "available_at"] = panel["available_at"].max() + pd.Timedelta(
        days=1
    )
    result = risk_features(changed)
    stock = result.loc[result["ticker"] == "AAPL"].iloc[-1]
    assert pd.isna(stock["beta_252d"])
    assert pd.isna(stock["idio_volatility_252d"])
    assert pd.notna(stock["volatility_60d"])


def test_unavailable_stock_prices_are_excluded(panel: pd.DataFrame) -> None:
    dates = panel.loc[panel["ticker"] == "AAPL", "date"].reset_index(drop=True)
    changed = panel.copy()
    mask = changed["ticker"].eq("AAPL") & changed["date"].isin(dates.iloc[251:264:3])
    changed.loc[mask, "available_at"] = panel["available_at"].max() + pd.Timedelta(
        days=1
    )
    result = risk_features(changed)
    stock = result.loc[result["ticker"] == "AAPL"].iloc[-1]
    assert pd.isna(stock["volatility_21d"])


def test_constant_market_has_no_identifiable_beta(panel: pd.DataFrame) -> None:
    changed = panel.copy()
    changed.loc[changed["ticker"] == "SPY", "adjusted_close"] = 100.0
    result = risk_features(changed)
    assert result[["beta_252d", "idio_volatility_252d"]].isna().all().all()
    assert set(result.columns) == {"ticker", "date", "available_at", *FEATURES}


def test_missing_market_is_an_explicit_error(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing market proxy SPY"):
        risk_features(panel.loc[panel["ticker"] != "SPY"])
