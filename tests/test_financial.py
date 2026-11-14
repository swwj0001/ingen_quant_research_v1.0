import math

import numpy as np
import pandas as pd
import pytest

from ingen_qr.features.financial import financial_features, trailing_amounts


def facts_fixture() -> pd.DataFrame:
    rows = []
    current = {
        "Assets": 1000,
        "StockholdersEquity": 500,
        "Liabilities": 500,
        "NetIncomeLoss": 120,
        "Revenues": 2000,
        "GrossProfit": 800,
        "CostOfRevenue": 1200,
        "NetCashProvidedByUsedInOperatingActivities": 160,
        "PaymentsToAcquirePropertyPlantAndEquipment": 40,
    }
    previous = {
        "Assets": 800,
        "StockholdersEquity": 400,
        "Liabilities": 400,
        "NetIncomeLoss": 100,
        "Revenues": 1600,
        "GrossProfit": 600,
        "CostOfRevenue": 1000,
        "NetCashProvidedByUsedInOperatingActivities": 100,
        "PaymentsToAcquirePropertyPlantAndEquipment": 30,
    }
    for year, values in [(2021, previous), (2022, current)]:
        for tag, value in values.items():
            rows.append(
                {
                    "cik": 1,
                    "namespace": "us-gaap",
                    "tag": tag,
                    "unit": "USD",
                    "value": float(value),
                    "start": (
                        None
                        if tag in ["Assets", "StockholdersEquity", "Liabilities"]
                        else f"{year}-01-01"
                    ),
                    "end": f"{year}-12-31",
                    "available_at": f"{year+1}-01-10T14:30:00Z",
                    "accn": str(year),
                    "frame": "DO_NOT_USE",
                }
            )
    rows.append(
        {
            "cik": 1,
            "namespace": "dei",
            "tag": "EntityCommonStockSharesOutstanding",
            "unit": "shares",
            "value": 20.0,
            "start": None,
            "end": "2022-12-31",
            "available_at": "2023-01-10T14:30:00Z",
            "accn": "2022",
            "frame": None,
        }
    )
    result = pd.DataFrame(rows)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    return result


def price_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["TEST"],
            "date": pd.to_datetime(["2023-01-31"]),
            "available_at": pd.to_datetime(["2023-01-31T21:30:00Z"]),
            "raw_close": [50.0],
            "split_ratio": [1.0],
        }
    )


EXPECTED = {
    "log_market_cap": math.log(1000),
    "book_to_market": 0.5,
    "earnings_yield": 0.12,
    "sales_to_price": 2,
    "cashflow_yield": 0.16,
    "roe": 120 / 450,
    "roa": 120 / 900,
    "gross_profitability": 0.8,
    "gross_margin": 0.4,
    "net_margin": 0.06,
    "operating_cashflow_to_assets": 0.16,
    "accruals": -40 / 900,
    "asset_growth": 0.25,
    "equity_growth": 0.25,
    "sales_growth": 0.25,
    "earnings_growth": 0.2,
    "liabilities_to_assets": 0.5,
    "equity_to_assets": 0.5,
    "liabilities_to_equity": 1,
    "capex_to_assets": 0.04,
    "capex_to_sales": 0.02,
    "free_cashflow_yield": 0.12,
    "asset_turnover": 2000 / 900,
    "log_assets": math.log(1000),
    "log_book_equity": math.log(500),
}


@pytest.mark.parametrize("feature,expected", EXPECTED.items())
def test_every_financial_feature_by_hand(feature: str, expected: float) -> None:
    values, _, _ = financial_features(
        facts_fixture(), price_fixture(), "TEST", pd.Timestamp("2023-01-31T22:00:00Z")
    )
    assert values[feature] == pytest.approx(expected)


def test_ttm_differences_ytd_and_reconstructs_q4() -> None:
    rows = pd.DataFrame(
        {
            "start": pd.to_datetime(["2021-01-01"] * 4 + ["2022-01-01"]),
            "end": pd.to_datetime(
                ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31", "2022-03-31"]
            ),
            "value": [10, 30, 60, 100, 15],
        }
    )
    totals = trailing_amounts(rows)
    assert totals[pd.Timestamp("2022-03-31")] == 105
    # Missing the Q3 cumulative amount prevents an invented Q4/TTM.
    assert pd.Timestamp("2022-03-31") not in trailing_amounts(rows.drop(index=2))


def test_direct_quarters_take_priority_over_duplicate_ytd_differences() -> None:
    rows = pd.DataFrame(
        {
            "start": pd.to_datetime(
                ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01", "2022-01-01"]
            ),
            "end": pd.to_datetime(
                ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2022-06-30"]
            ),
            "value": [10, 20, 30, 40, 31],
        }
    )
    assert trailing_amounts(rows)[pd.Timestamp("2022-12-31")] == 100


def test_future_filing_ifrs_and_frame_cannot_change_past_features() -> None:
    original, prices = facts_fixture(), price_fixture()
    first = financial_features(
        original, prices, "TEST", pd.Timestamp("2023-01-31T22:00:00Z")
    )[0]
    changed = original.copy()
    changed["frame"] = "CY2099Q1"
    future = original.copy()
    future["available_at"] = pd.Timestamp("2024-01-02T14:30:00Z")
    future["value"] *= 100
    foreign = original.copy()
    foreign["namespace"], foreign["value"] = "ifrs-full", 123456.0
    second = financial_features(
        pd.concat([changed, future, foreign]),
        prices,
        "TEST",
        pd.Timestamp("2023-01-31T22:00:00Z"),
    )[0]
    assert first == second


def test_shares_priority_splits_and_ambiguous_classes() -> None:
    facts, prices = facts_fixture(), price_fixture()
    fallback = facts.iloc[-1:].copy()
    fallback["namespace"], fallback["tag"], fallback["value"] = (
        "us-gaap",
        "CommonStockSharesOutstanding",
        999.0,
    )
    prices["raw_close"], prices["split_ratio"] = 25.0, 2.0
    values, _, metadata = financial_features(
        pd.concat([facts, fallback]),
        prices,
        "TEST",
        pd.Timestamp("2023-01-31T22:00:00Z"),
    )
    assert metadata["shares_outstanding"] == 40
    assert values["log_market_cap"] == pytest.approx(math.log(1000))
    ambiguous = financial_features(
        facts,
        prices,
        "TEST",
        pd.Timestamp("2023-01-31T22:00:00Z"),
        ambiguous_class=True,
    )[0]
    assert np.isnan(ambiguous["book_to_market"])


def test_unreported_flags_do_not_encode_missing_price_history() -> None:
    facts = facts_fixture()
    prices = price_fixture().iloc[:0]
    values, flags, _ = financial_features(
        facts, prices, "TEST", pd.Timestamp("2023-01-31T22:00:00Z")
    )
    assert np.isnan(values["log_market_cap"])
    assert not flags["log_market_cap"]
    missing = facts.loc[facts["tag"] != "NetIncomeLoss"]
    assert financial_features(
        missing, prices, "TEST", pd.Timestamp("2023-01-31T22:00:00Z")
    )[1]["earnings_yield"]


def test_inputs_unchanged_and_gross_profit_can_use_aligned_revenue_minus_cost() -> None:
    facts, prices = facts_fixture(), price_fixture()
    saved = facts.copy(deep=True)
    without_gross = facts.loc[facts["tag"] != "GrossProfit"]
    values = financial_features(
        without_gross, prices, "TEST", pd.Timestamp("2023-01-31T22:00:00Z")
    )[0]
    assert values["gross_profitability"] == 0.8
    pd.testing.assert_frame_equal(facts, saved)
