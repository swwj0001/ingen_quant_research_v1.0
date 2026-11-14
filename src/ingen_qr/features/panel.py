# mypy: disable-error-code=import-untyped

"""One historical constituent snapshot per month; explicit CIK exclusion."""

from pathlib import Path
from typing import Any

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ingen_qr.data.price_audit import recover_development_prices
from ingen_qr.data.storage import as_of
from ingen_qr.features.financial import FEATURES as FINANCIAL_FEATURES
from ingen_qr.features.financial import financial_features
from ingen_qr.features.liquidity import FEATURES as LIQUIDITY_FEATURES
from ingen_qr.features.liquidity import liquidity_features
from ingen_qr.features.macro import FEATURES as MACRO_FEATURES
from ingen_qr.features.macro import (
    INTERACTIONS,
    first_vintage_market,
    macro_snapshot,
)
from ingen_qr.features.momentum import FEATURES as MOMENTUM_FEATURES
from ingen_qr.features.momentum import momentum_features
from ingen_qr.features.risk import FEATURES as RISK_FEATURES
from ingen_qr.features.risk import risk_features

TURNOVER_FEATURES = ["turnover_21d", "turnover_60d"]
FIRM_FEATURES = [
    *MOMENTUM_FEATURES,
    *RISK_FEATURES,
    *LIQUIDITY_FEATURES,
    *FINANCIAL_FEATURES,
    *TURNOVER_FEATURES,
]
FEATURES = [*FIRM_FEATURES, *INTERACTIONS]


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    result = frame.copy()
    for column in result:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.as_unit("us")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)


def monthly_turnover(
    stock: pd.DataFrame,
    origin: pd.Timestamp,
    cutoff: pd.Timestamp,
    shares: float,
    sessions: pd.DatetimeIndex,
) -> dict[str, float]:
    known = stock.loc[
        (stock["date"] <= origin) & (stock["available_at"] <= cutoff)
    ].set_index("date")
    grid = sessions[sessions <= origin][-60:]
    data = known.reindex(grid)
    splits = data["split_ratio"].replace(0.0, 1.0)
    # Move historical volume into the same split units as as-of shares.
    future = splits.iloc[::-1].cumprod(skipna=False).iloc[::-1] / splits
    normalized = (
        data["volume"] * future / shares
        if shares > 0 and np.isfinite(shares)
        else data["volume"] * np.nan
    )
    return {
        f"turnover_{width}d": (
            float(normalized.iloc[-width:].mean())
            if normalized.iloc[-width:].notna().sum() >= (4 * width + 4) // 5
            else np.nan
        )
        for width in (21, 60)
    }


def build_monthly_panel(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    output: Path,
    allow_network: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = str(config["development_start"]), str(config["development_end"])
    if pd.Timestamp(end) >= pd.Timestamp("2024-01-01"):
        raise PermissionError(
            "This W2 command only builds the sealed development panel."
        )
    connection.execute("SET TimeZone='UTC'")
    cohorts = connection.execute(
        "SELECT * FROM universe WHERE date BETWEEN ?::DATE AND ?::DATE ORDER BY date, ticker",
        [start, end],
    ).df()
    if cohorts.empty:
        raise ValueError("No development cohorts in the database.")
    cohorts["date"] = pd.to_datetime(cohorts["date"]).dt.tz_localize(None)
    cohorts["forecast_at"] = pd.to_datetime(cohorts["forecast_at"], utc=True)
    if cohorts.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate historical cohort rows.")
    if "cik" not in cohorts:
        raise ValueError(
            "The universe view must contain the existing ticker-to-CIK mapping."
        )
    excluded = cohorts.groupby("date").agg(
        original_pool=("ticker", "size"), mapped_cik=("cik", "count")
    )
    excluded["no_cik_excluded"] = excluded["original_pool"] - excluded["mapped_cik"]
    excluded["no_cik_exclusion_share"] = (
        excluded["no_cik_excluded"] / excluded["original_pool"]
    )
    excluded.reset_index().to_csv(output / "cik_exclusions.csv", index=False)
    cohorts = cohorts.loc[cohorts["cik"].notna()].copy()
    if cohorts.empty:
        raise ValueError(
            "No CIK-mapped companies remain under the fixed eligibility rule."
        )
    cohorts["cik"] = cohorts["cik"].astype("int64")
    prices = connection.execute(
        "SELECT * FROM prices WHERE date <= ?::DATE", [end]
    ).df()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices["available_at"] = pd.to_datetime(prices["available_at"], utc=True)
    prices, recovery = recover_development_prices(
        prices, Path("data/reports"), Path("data/raw/prices"), end, allow_network
    )
    recovery.to_csv(output / "price_recovery.csv", index=False)
    save_parquet(prices, output.parent / "prices_development.parquet")
    needed = set(cohorts["ticker"]) | {str(config["market_proxy"])}
    prices = prices.loc[prices["ticker"].isin(needed)].copy()
    calendar = xcals.get_calendar(
        "XNYS",
        start=prices["date"].min(),
        end=pd.Timestamp(end) + pd.Timedelta(days=15),
    )
    origins = {day: calendar.previous_session(day) for day in cohorts["date"].unique()}
    origin_dates = set(origins.values())
    print(
        f"Prices: {prices['ticker'].nunique()} tickers; calculating the existing 11 daily features.",
        flush=True,
    )
    daily = momentum_features(prices)
    daily = daily.loc[daily["date"].isin(origin_dates)].copy()
    risk = risk_features(prices, str(config["market_proxy"]))
    daily = daily.merge(
        risk.loc[risk["date"].isin(origin_dates)],
        on=["ticker", "date", "available_at"],
        validate="one_to_one",
    )
    del risk
    liquidity = liquidity_features(prices)
    daily = daily.merge(
        liquidity.loc[liquidity["date"].isin(origin_dates)],
        on=["ticker", "date", "available_at"],
        validate="one_to_one",
    )
    del liquidity
    daily_index = daily.set_index(["ticker", "date"]).to_dict("index")
    price_groups = {
        str(ticker): block.copy() for ticker, block in prices.groupby("ticker")
    }
    fred = connection.execute("SELECT * FROM fred WHERE date <= ?::DATE", [end]).df()
    market, revisions = first_vintage_market(fred)
    revisions.to_csv(output / "macro_revision_audit.csv", index=False)
    save_parquet(market, output.parent / "market_first_vintage.parquet")
    records, macro_records = [], []
    for number, (date, cohort) in enumerate(cohorts.groupby("date", sort=True), 1):
        date = pd.Timestamp(str(date))
        cutoff = cohort["forecast_at"].iloc[0]
        if cohort["forecast_at"].nunique() != 1:
            raise ValueError("A monthly cohort has multiple forecast cutoffs.")
        origin = origins[date]
        print(
            f"Month {number}: {date.date()} ({len(cohort)} CIK-mapped stocks)",
            flush=True,
        )
        facts = as_of(connection, "company_facts", cutoff.isoformat())
        fact_groups = {int(str(cik)): group for cik, group in facts.groupby("cik")}
        macro_values, macro_details = macro_snapshot(market, cutoff)
        macro_records.append(
            {"date": date, "forecast_at": cutoff, **macro_values, **macro_details}
        )
        classes = cohort.groupby("cik")["ticker"].nunique().to_dict()
        for row in cohort.to_dict("records"):
            values, flags, details = financial_features(
                fact_groups.get(int(row["cik"]), facts.iloc[:0]),
                price_groups.get(str(row["ticker"]), prices.iloc[:0]),
                str(row["ticker"]),
                cutoff,
                int(config["financial_max_age_days"]),
                bool(classes[row["cik"]] > 1),
            )
            record: dict[str, Any] = {
                "date": date,
                "ticker": str(row["ticker"]),
                "cik": row["cik"],
                "forecast_at": cutoff,
                "available_at": cutoff,
                "origin_date": origin,
                **values,
                **macro_values,
                **details,
            }
            key = (str(row["ticker"]), origin)
            price_known = (
                key in daily_index and daily_index[key]["available_at"] <= cutoff
            )
            record["price_at_forecast"] = price_known
            for name in [*MOMENTUM_FEATURES, *RISK_FEATURES, *LIQUIDITY_FEATURES]:
                record[name] = daily_index[key][name] if price_known else np.nan
            if not price_known:
                for name in (
                    "log_market_cap",
                    "book_to_market",
                    "earnings_yield",
                    "sales_to_price",
                    "cashflow_yield",
                    "free_cashflow_yield",
                ):
                    record[name] = np.nan
            stock = price_groups.get(str(row["ticker"]), prices.iloc[:0])
            record.update(
                monthly_turnover(
                    stock,
                    origin,
                    cutoff,
                    float(details["shares_outstanding"]),
                    pd.DatetimeIndex(calendar.sessions),
                )
            )
            record.update({"unreported_" + name: flag for name, flag in flags.items()})
            records.append(record)
    panel = pd.DataFrame(records)
    conditions = pd.DataFrame(macro_records)
    conditions["available_at"] = conditions["forecast_at"]
    conditions.to_csv(output / "macro_staleness.csv", index=False)
    save_parquet(conditions, output.parent / "macro_conditions.parquet")
    condition_coverage = conditions[["date", *MACRO_FEATURES]].melt(
        id_vars="date", var_name="condition", value_name="value"
    )
    condition_coverage["observed"] = np.isfinite(condition_coverage["value"])
    condition_coverage.to_csv(output / "macro_condition_coverage.csv", index=False)
    counts = panel.groupby("date").agg(
        research_pool=("ticker", "size"),
        forecast_price_covered=("price_at_forecast", "sum"),
    )
    counts.to_csv(output / "research_pool_coverage.csv")
    return panel, cohorts, market
