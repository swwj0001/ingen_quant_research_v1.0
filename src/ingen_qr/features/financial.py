"""Availability-filtered US-GAAP/dei fundamentals; no SEC frame selection."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pandas as pd

TAGS = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity"],
    "liabilities": ["Liabilities"],
    "income": ["NetIncomeLoss"],
    "sales": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross": ["GrossProfit"],
    "cost": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
SHARES = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]
FEATURES = [
    "log_market_cap",
    "book_to_market",
    "earnings_yield",
    "sales_to_price",
    "cashflow_yield",
    "roe",
    "roa",
    "gross_profitability",
    "gross_margin",
    "net_margin",
    "operating_cashflow_to_assets",
    "accruals",
    "asset_growth",
    "equity_growth",
    "sales_growth",
    "earnings_growth",
    "liabilities_to_assets",
    "equity_to_assets",
    "liabilities_to_equity",
    "capex_to_assets",
    "capex_to_sales",
    "free_cashflow_yield",
    "asset_turnover",
    "log_assets",
    "log_book_equity",
]
REQUIRES = {
    "log_market_cap": ["shares"],
    "book_to_market": ["equity", "shares"],
    "earnings_yield": ["income", "shares"],
    "sales_to_price": ["sales", "shares"],
    "cashflow_yield": ["cashflow", "shares"],
    "roe": ["income", "equity"],
    "roa": ["income", "assets"],
    "gross_profitability": ["gross", "assets"],
    "gross_margin": ["gross", "sales"],
    "net_margin": ["income", "sales"],
    "operating_cashflow_to_assets": ["cashflow", "assets"],
    "accruals": ["income", "cashflow", "assets"],
    "asset_growth": ["assets"],
    "equity_growth": ["equity"],
    "sales_growth": ["sales"],
    "earnings_growth": ["income"],
    "liabilities_to_assets": ["liabilities", "assets"],
    "equity_to_assets": ["equity", "assets"],
    "liabilities_to_equity": ["liabilities", "equity"],
    "capex_to_assets": ["capex", "assets"],
    "capex_to_sales": ["capex", "sales"],
    "free_cashflow_yield": ["cashflow", "capex", "shares"],
    "asset_turnover": ["sales", "assets"],
    "log_assets": ["assets"],
    "log_book_equity": ["equity"],
}


@dataclass(frozen=True)
class Amount:
    value: float
    end: pd.Timestamp


def divide(numerator: float, denominator: float) -> float:
    return (
        numerator / denominator
        if np.isfinite(denominator) and denominator > 0
        else float("nan")
    )


def logged(value: float) -> float:
    return float(np.log(value)) if value > 0 and np.isfinite(value) else float("nan")


def concept_rows(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Choose one available tag per start/end; never read frame/fy/fp."""
    tags = TAGS[concept]
    rows = facts.loc[
        (facts["namespace"] == "us-gaap")
        & facts["tag"].isin(tags)
        & (facts["unit"] == "USD")
    ].copy()
    rows["priority"] = rows["tag"].map({tag: i for i, tag in enumerate(tags)})
    return rows.sort_values(
        ["end", "start", "priority", "available_at", "accn"], na_position="first"
    ).drop_duplicates(["start", "end"], keep="first")


def instantaneous(rows: pd.DataFrame, end: pd.Timestamp) -> Amount | None:
    rows = rows.loc[rows["start"].isna() & (rows["end"] <= end)].sort_values("end")
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return Amount(float(row["value"]), pd.Timestamp(row["end"]))


def _trailing_records(records: list[dict[str, Any]]) -> dict[pd.Timestamp, float]:
    periods: dict[tuple[pd.Timestamp, pd.Timestamp], float] = {}
    totals: dict[pd.Timestamp, float] = {}
    groups: dict[pd.Timestamp, list[tuple[pd.Timestamp, float]]] = {}
    for row in records:
        if pd.isna(row["start"]):
            continue
        start, end, value = row["start"], row["end"], float(row["value"])
        days = (end - start).days + 1
        if not np.isfinite(value) or not 65 <= days <= 400:
            continue
        groups.setdefault(start, []).append((end, value))
        if days <= 110:
            periods[(start, end)] = value
        if days >= 330:
            totals[end] = value
    for cumulative in groups.values():
        ordered = sorted(cumulative)
        for (previous_end, previous_value), (end, value) in pairwise(ordered):
            if 65 <= (end - previous_end).days <= 110:
                periods.setdefault(
                    (previous_end + pd.Timedelta(days=1), end), value - previous_value
                )
    by_end = {end: (start, value) for (start, end), value in sorted(periods.items())}
    for end, (start, value) in by_end.items():
        total, first, complete = value, start, True
        for _ in range(3):
            prior = by_end.get(first - pd.Timedelta(days=1))
            if prior is None:
                complete = False
                break
            first, part = prior
            total += part
        if complete and 330 <= (end - first).days + 1 <= 400:
            totals.setdefault(end, total)
    return totals


def trailing_amounts(rows: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """Annual totals or four adjacent discrete quarters reconstructed from YTD."""
    return _trailing_records(cast(list[dict[str, Any]], rows.to_dict("records")))


def _statement_rows(rows: list[dict[str, Any]], concept: str) -> list[dict[str, Any]]:
    tags = TAGS[concept]
    selected: dict[tuple[object, object], dict[str, Any]] = {}
    for row in rows:
        if (
            row["namespace"] != "us-gaap"
            or row["unit"] != "USD"
            or row["tag"] not in tags
        ):
            continue
        key = (row["start"] if pd.notna(row["start"]) else None, row["end"])
        old = selected.get(key)
        if (
            old is None
            or tags.index(row["tag"]) < tags.index(old["tag"])
            or (
                row["tag"] == old["tag"]
                and (row["available_at"], row["accn"])
                > (old["available_at"], old["accn"])
            )
        ):
            selected[key] = row
    return list(selected.values())


def latest_total(totals: dict[pd.Timestamp, float], end: pd.Timestamp) -> Amount | None:
    dates = [day for day in totals if day <= end]
    if not dates:
        return None
    last = max(dates)
    return Amount(totals[last], last)


def _previous(item: Amount | None, values: dict[pd.Timestamp, float]) -> float:
    if item is None:
        return float("nan")
    target = item.end - pd.DateOffset(years=1)
    candidates = [day for day in values if abs((day - target).days) <= 30]
    if not candidates:
        return float("nan")
    return values[min(candidates, key=lambda day: abs((day - target).days))]


def financial_features(
    facts: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
    cutoff: pd.Timestamp,
    max_age_days: int = 550,
    ambiguous_class: bool = False,
) -> tuple[dict[str, float], dict[str, bool], dict[str, Any]]:
    """Caller must supply storage.as_of output; enforce availability again."""
    rows = facts.loc[
        facts["namespace"].isin(["us-gaap", "dei"]) & (facts["available_at"] <= cutoff)
    ].copy()
    for column in ("start", "end"):
        rows[column] = pd.to_datetime(rows[column])
    end = cutoff.tz_convert("America/New_York").tz_localize(None).normalize()
    current: dict[str, float] = {}
    previous: dict[str, float] = {}
    present: dict[str, bool] = {}
    ages: list[int] = []
    records = cast(list[dict[str, Any]], rows.to_dict("records"))
    selected_by_concept = {
        concept: _statement_rows(records, concept) for concept in TAGS
    }
    for concept, selected in selected_by_concept.items():
        present[concept] = bool(selected)
        if concept in ("assets", "equity", "liabilities"):
            history = {
                r["end"]: float(r["value"])
                for r in selected
                if pd.isna(r["start"]) and r["end"] <= end
            }
        else:
            history = _trailing_records(selected)
        item = latest_total(history, end)
        current[concept] = float("nan")
        if item is not None:
            age = (end - item.end).days
            ages.append(age)
            if age <= max_age_days:
                current[concept] = item.value
        previous[concept] = _previous(item, history)
    if not present["gross"] and present["sales"] and present["cost"]:
        sales_totals = _trailing_records(selected_by_concept["sales"])
        cost_totals = _trailing_records(selected_by_concept["cost"])
        common = {
            day: sales_totals[day] - cost_totals[day]
            for day in sales_totals.keys() & cost_totals.keys()
        }
        gross = latest_total(common, end)
        if gross is not None and (end - gross.end).days <= max_age_days:
            current["gross"] = gross.value
        present["gross"] = True
    shares = float("nan")
    share_end = pd.NaT
    present["shares"] = False
    for namespace, tag in SHARES:
        candidates = rows.loc[
            (rows["namespace"] == namespace)
            & (rows["tag"] == tag)
            & (rows["unit"] == "shares")
            & rows["start"].isna()
            & (rows["end"] <= end)
        ].sort_values("end")
        present["shares"] |= not candidates.empty
        if not candidates.empty:
            selected_share = candidates.iloc[-1]
            share_end = selected_share["end"]
            if (end - share_end).days <= max_age_days:
                shares = float(selected_share["value"])
                break
    stock = prices.loc[
        (prices["ticker"] == ticker)
        & (prices["available_at"] <= cutoff)
        & (prices["date"] <= end)
    ].sort_values("date")
    price = float("nan")
    if not stock.empty:
        if stock.iloc[-1]["date"] == end:
            price = float(stock.iloc[-1]["raw_close"])
        if pd.notna(share_end):
            split_rows = stock.loc[stock["date"] > share_end, "split_ratio"]
            shares *= float(split_rows.replace(0.0, 1.0).prod())
    market_cap = price * shares if not ambiguous_class else float("nan")
    a, e, l = (current[c] for c in ("assets", "equity", "liabilities"))
    n, s, g, c, x = (
        current[k] for k in ("income", "sales", "gross", "cashflow", "capex")
    )
    avg_a, avg_e = (a + previous["assets"]) / 2, (e + previous["equity"]) / 2
    values = [
        logged(market_cap),
        divide(e if e > 0 else float("nan"), market_cap),
        divide(n, market_cap),
        divide(s, market_cap),
        divide(c, market_cap),
        divide(n, avg_e),
        divide(n, avg_a),
        divide(g, a),
        divide(g, s),
        divide(n, s),
        divide(c, a),
        divide(n - c, avg_a),
        divide(a, previous["assets"]) - 1,
        divide(e, previous["equity"]) - 1,
        divide(s, previous["sales"]) - 1,
        divide(n - previous["income"], abs(previous["income"])),
        divide(l, a),
        divide(e, a),
        divide(l, e),
        divide(x, a),
        divide(x, s),
        divide(c - x, market_cap),
        divide(s, avg_a),
        logged(a),
        logged(e),
    ]
    features = dict(zip(FEATURES, values, strict=True))
    flags = {
        name: bool(not rows.empty and any(not present[c] for c in REQUIRES[name]))
        for name in FEATURES
    }
    details: dict[str, Any] = {
        "shares_outstanding": shares,
        "shares_date": share_end,
        "financial_max_age_days": max(ages) if ages else np.nan,
        "ambiguous_share_class": ambiguous_class,
    }
    return features, flags, details
