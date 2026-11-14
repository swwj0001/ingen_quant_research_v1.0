import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ingen_qr.evaluation.labels import make_labels
from ingen_qr.evaluation.splitters import PurgedWalkForward


def label_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = xcals.get_calendar("XNYS", start="2023-01-01", end="2023-04-30")
    entry = pd.Timestamp("2023-02-01")
    dates = calendar.sessions_in_range(entry, "2023-03-31")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "adjusted_open": 100.0,
            "adjusted_close": 110.0,
            "available_at": [
                calendar.session_close(d) + pd.Timedelta(minutes=30) for d in dates
            ],
        }
    )
    cohorts = pd.DataFrame(
        {
            "date": [entry],
            "ticker": ["A"],
            "forecast_at": pd.to_datetime(["2023-01-31T22:00:00Z"]),
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-30", "2023-02-01"]),
            "series_id": ["DGS3MO", "DGS3MO"],
            "value": [0.0365, 0.99],
            "available_at": pd.to_datetime(
                ["2023-01-31T14:30:00Z", "2023-02-02T14:30:00Z"]
            ),
        }
    )
    return cohorts, prices, market


def test_label_21_sessions_open_to_close_and_forecast_known_rf() -> None:
    cohorts, prices, market = label_fixture()
    result = make_labels(cohorts, prices, market)
    assert result["label_end"].item() == prices["date"].iloc[20]
    calendar = xcals.get_calendar("XNYS", start="2023-01-01", end="2023-04-30")
    days = (
        calendar.session_close(prices["date"].iloc[20])
        - calendar.session_open(cohorts["date"].item())
    ).total_seconds() / 86400
    assert result["risk_free_21d"].item() == pytest.approx(0.0365 * days / 365)
    assert result["label_excess_return"].item() == pytest.approx(
        0.1 - 0.0365 * days / 365
    )
    changed = prices.copy()
    changed.loc[changed.index > 20, "adjusted_close"] *= 100
    pd.testing.assert_frame_equal(result, make_labels(cohorts, changed, market))


def test_tercile_middle_and_missing_terminal_price_are_not_fabricated() -> None:
    cohorts, prices, market = label_fixture()
    pieces, pools = [], []
    for i, ticker in enumerate(["A", "B", "C", "D", "E", "F"]):
        piece, pool = prices.copy(), cohorts.copy()
        piece["ticker"], pool["ticker"] = ticker, ticker
        piece["adjusted_close"] += i
        pieces.append(piece)
        pools.append(pool)
    result = make_labels(pd.concat(pools), pd.concat(pieces), market)
    assert result["label_class"].tolist()[:2] == [0, 0]
    assert result["label_class"].isna().sum() == 2
    assert result["label_class"].tolist()[-2:] == [1, 1]
    absent = make_labels(cohorts, prices.drop(index=20), market)
    assert np.isnan(absent["label_excess_return"].item())


def test_labels_and_splitter_refuse_holdout_until_explicit_unlock() -> None:
    cohorts, prices, market = label_fixture()
    with pytest.raises(PermissionError):
        make_labels(cohorts, prices, market, end="2024-12-31")
    with pytest.raises(PermissionError):
        PurgedWalkForward(last_test="2024-01-01")
    PurgedWalkForward(last_test="2024-01-01", unlock_holdout=True)


@settings(max_examples=35, deadline=None)
@given(
    horizon=st.integers(min_value=1, max_value=70),
    embargo=st.integers(min_value=0, max_value=3),
    duplicates=st.integers(min_value=1, max_value=4),
)
def test_property_no_overlap_no_holdout_and_no_duplicate_test_month(
    horizon: int, embargo: int, duplicates: int
) -> None:
    dates = pd.date_range("2010-01-01", "2023-09-01", freq="MS").repeat(duplicates)
    ends = dates + pd.to_timedelta(horizon, unit="D")
    meta = pd.DataFrame(
        {
            "date": dates,
            "label_end": ends,
            "label_available_at": pd.to_datetime(ends, utc=True)
            + pd.Timedelta(hours=22),
            "forecast_at": pd.to_datetime(dates, utc=True) - pd.Timedelta(days=1),
        }
    )
    seen = []
    for fold in PurgedWalkForward(last_test="2023-09-01", embargo_months=embargo).split(
        meta
    ):
        boundary = fold.test_month - pd.DateOffset(months=12 + embargo)
        assert (meta.iloc[fold.train]["label_end"] < boundary).all()
        assert (
            meta.iloc[fold.refit]["label_end"]
            < fold.test_month - pd.DateOffset(months=embargo)
        ).all()
        assert set(fold.train).isdisjoint(set(fold.validation))
        assert set(fold.refit).isdisjoint(set(fold.test))
        assert (meta.iloc[fold.test]["label_end"] < pd.Timestamp("2024-01-01")).all()
        seen.extend(fold.test.tolist())
    assert len(seen) == len(set(seen))
