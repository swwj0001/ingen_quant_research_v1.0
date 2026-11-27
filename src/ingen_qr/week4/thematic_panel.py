# mypy: disable-error-code=import-untyped

"""Build the thematic monthly universe with the existing feature and label code."""

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
from yfinance.exceptions import YFTickerMissingError

from ingen_qr.data.download import Downloader, TemporaryDownloadError
from ingen_qr.data.price_panel import _prepare_price_panel
from ingen_qr.data.prices import load_price_source
from ingen_qr.data.sec import normalize_company_facts
from ingen_qr.evaluation.benchmarks import build_benchmark_sample
from ingen_qr.evaluation.labels import make_labels
from ingen_qr.evaluation.week3_samples import file_sha256
from ingen_qr.features.financial import FEATURES as FUNDAMENTALS
from ingen_qr.features.panel import FIRM_FEATURES, build_monthly_panel
from ingen_qr.features.preprocessing import preprocess_panel
from ingen_qr.week4.baseline import sample
from ingen_qr.week4.common import (
    KEYS,
    WEEK2,
    config,
    parquet,
    require_baseline,
    write_json,
)
from ingen_qr.week4.filings import sec_client

MODELED_SECURITIES = {"us_equity", "us_listed_foreign_equity", "us_adr"}


def load_candidates(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path, keep_default_na=False)
    if candidates["ticker"].duplicated().any() or not 40 <= len(candidates) <= 60:
        raise ValueError("The thematic coverage table needs 40–60 unique tickers.")
    if not set(candidates["security"]) <= MODELED_SECURITIES | {"coverage_only"}:
        raise ValueError("Unknown thematic security type.")
    us_listed = candidates["security"].isin(MODELED_SECURITIES)
    if (
        not candidates.loc[us_listed, "market"]
        .isin(["NYSE", "Nasdaq", "NYSE American"])
        .all()
    ):
        raise ValueError(
            "All modeled thematic equities and ADRs need a US exchange listing."
        )
    return candidates


def source_prices(
    candidates: pd.DataFrame, frozen: Path, directory: Path, market: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    universe = candidates["ticker"].tolist()
    prices = pd.read_parquet(
        frozen / "prices_development.parquet",
        filters=[("ticker", "in", [*universe, market])],
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices["available_at"] = pd.to_datetime(prices["available_at"], utc=True)
    prices = prices.loc[prices["date"].lt("2024-01-01")]
    status = {
        str(ticker): "frozen_development_prices" for ticker in prices["ticker"].unique()
    }
    frames = [prices]
    client = Downloader(
        str(directory / "price_cache"), min_interval=1.0, max_attempts=3
    )
    try:
        for ticker in universe:
            if ticker in status:
                continue
            try:
                source = load_price_source(ticker, "2008-01-01", "2024-01-01", client)
                prepared = _prepare_price_panel(source, ticker)
                frames.append(prepared)
                status[ticker] = "bounded_yahoo_history"
            except (
                ValueError,
                KeyError,
                YFTickerMissingError,
                TemporaryDownloadError,
                requests.RequestException,
            ) as error:
                status[ticker] = type(error).__name__ + ": " + str(error)
    finally:
        client.session.close()
    combined = pd.concat(frames, ignore_index=True)
    for row in candidates.itertuples(index=False):
        outside = combined["ticker"].eq(str(row.ticker))
        if row.listed_from:
            combined = combined.loc[~(outside & combined["date"].lt(row.listed_from))]
            outside = combined["ticker"].eq(str(row.ticker))
        if row.listed_until:
            combined = combined.loc[~(outside & combined["date"].gt(row.listed_until))]
    return combined, status


def thematic_cohorts(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    status: dict[str, str],
    dates: pd.DataFrame,
    ciks: dict[str, int],
    short_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage, entries = [], []
    for candidate in candidates.to_dict("records"):
        ticker = candidate["ticker"]
        group = prices.loc[prices["ticker"].eq(ticker)].sort_values("date")
        first_date, last_date = group["date"].min(), group["date"].max()
        last_listed = (
            pd.Timestamp(candidate["listed_until"])
            if candidate["listed_until"]
            else None
        )
        first_listed = (
            pd.Timestamp(candidate["listed_from"])
            if candidate["listed_from"]
            else first_date
        )
        effective = (
            max(first_listed, first_date) if pd.notna(first_date) else first_listed
        )
        modeling = candidate["security"] != "coverage_only"
        records = []
        eligible_calendar_months = 0
        for row in dates.to_dict("records"):
            if not modeling or pd.isna(effective) or row["date"] < effective:
                continue
            if last_listed is not None and row["date"] > last_listed:
                continue
            eligible_calendar_months += 1
            if int(group["available_at"].le(row["forecast_at"]).sum()) == 0:
                continue
            records.append(
                {
                    **row,
                    "ticker": ticker,
                    "cik": ciks.get(ticker),
                    "market": candidate["market"],
                    "available_at": row["forecast_at"],
                    "history_sessions": int(
                        group["available_at"].le(row["forecast_at"]).sum()
                    ),
                }
            )
        entries.extend(records)
        coverage.append(
            {
                **candidate,
                "modeled_security_type": modeling,
                "observed_first_quote": first_date,
                "observed_last_quote": last_date,
                "effective_listing_start": effective,
                "eligible_calendar_months": eligible_calendar_months,
                "candidate_months": len(records),
                "short_history_months": sum(
                    r["history_sessions"] < short_history for r in records
                ),
                "price_status": status.get(ticker, "retained_for_coverage_only"),
                "cik": ciks.get(ticker),
            }
        )
    result = pd.DataFrame(entries)
    if result.empty:
        raise ValueError(
            "No eligible thematic ticker-months. Inspect price data and listing dates."
        )
    return result, pd.DataFrame(coverage)


def build(settings: dict[str, Any], text: dict[str, Any], directory: Path) -> None:
    gate = require_baseline(Path(text["output_directory"]))
    sample()
    settings2 = config(WEEK2)
    frozen = Path(settings2["output_directory"])
    if settings2["development_end"] != "2023-12-31":
        raise ValueError("Thematic development stops before the sealed holdout.")
    candidates = load_candidates(Path(settings["candidates"]))
    model_candidates = candidates.loc[candidates["security"].ne("coverage_only")]
    directory.mkdir(parents=True, exist_ok=True)
    reports = directory / "reports"
    reports.mkdir(exist_ok=True)
    db = Path(settings2["database"])
    if not db.is_file():
        raise FileNotFoundError(f"Week 2 development database is missing: {db}")
    prices, price_status = source_prices(
        model_candidates, frozen, directory, settings2["market_proxy"]
    )
    dates = pd.read_parquet(
        frozen / "features_raw.parquet", columns=["date", "forecast_at"]
    ).drop_duplicates()
    dates["forecast_at"] = pd.to_datetime(dates["forecast_at"], utc=True)
    if dates["date"].duplicated().any() or dates["date"].ge("2024-01-01").any():
        raise ValueError(
            "Thematic cutoffs must match the sealed monthly Week 2 schedule."
        )
    raw = pd.read_parquet(frozen / "features_raw.parquet", columns=["ticker", "cik"])
    ciks = {
        str(t): int(cik)
        for t, cik in raw.dropna().drop_duplicates().itertuples(index=False)
    }
    filings = {}
    client = sec_client(text, str(Path(text["output_directory"]) / "cache" / "sec"))
    try:
        ticker_snapshot = Path("data/raw/sec/company_tickers.json")
        body = (
            ticker_snapshot.read_bytes()
            if ticker_snapshot.is_file()
            else client.get("https://www.sec.gov/files/company_tickers.json")
        )
        mapped = {
            str(row["ticker"]).upper(): int(row["cik_str"])
            for row in json.loads(body).values()
        }
        for ticker in model_candidates["ticker"]:
            if ticker in mapped:
                ciks.setdefault(ticker, mapped[ticker])
        cohorts, coverage = thematic_cohorts(
            candidates,
            prices,
            price_status,
            dates.sort_values("date"),
            ciks,
            int(settings["short_history_sessions"]),
        )
        needed = sorted({ciks[t] for t in cohorts["ticker"] if t in ciks})
        with duckdb.connect(str(db), read_only=True) as existing:
            facts = existing.execute(
                "SELECT * FROM company_facts WHERE cik IN (SELECT unnest(?::BIGINT[])) "
                "AND available_at < TIMESTAMPTZ '2024-01-01 00:00:00+00'",
                [needed],
            ).df()
            fred = existing.execute(
                "SELECT * FROM fred WHERE date <= DATE '2023-12-31'"
            ).df()
        local = set(facts["cik"].astype(int))
        filings.update({cik: "frozen_company_facts" for cik in local})
        pieces = [facts]
        tags = config(Path("configs/pipeline.yaml"))["sec_tags"]
        for cik in set(needed) - local:
            try:
                payload = json.loads(
                    client.get(
                        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
                    )
                )
                frame = normalize_company_facts(payload, tags)
                if not frame.empty:
                    available = pd.to_datetime(frame["available_at"], utc=True)
                    frame = frame.loc[
                        available.lt(pd.Timestamp("2024-01-01", tz="UTC"))
                    ]
                    pieces.append(frame)
                filings[cik] = (
                    "downloaded_development_company_facts"
                    if not frame.empty
                    else "no_development_xbrl"
                )
            except (
                requests.RequestException,
                TemporaryDownloadError,
                ValueError,
                KeyError,
            ) as error:
                filings[cik] = type(error).__name__ + ": " + str(error)
    finally:
        client.session.close()
    other = coverage["security"].eq("coverage_only")
    quotes = Downloader(
        str(directory / "price_cache"), min_interval=1.0, max_attempts=2
    )
    try:
        for idx, row in coverage.loc[other].iterrows():
            try:
                source = load_price_source(
                    str(row["ticker"]), "2008-01-01", "2024-01-01", quotes
                )
                observed = pd.DatetimeIndex(source.index).tz_localize(None).normalize()
                coverage.loc[idx, ["observed_first_quote", "observed_last_quote"]] = [
                    observed.min(),
                    observed.max(),
                ]
                coverage.loc[idx, "effective_listing_start"] = (
                    max(pd.Timestamp(row["listed_from"]), observed.min())
                    if row["listed_from"]
                    else observed.min()
                )
                coverage.loc[idx, "price_status"] = "coverage_only_observed_quote"
            except (
                ValueError,
                KeyError,
                YFTickerMissingError,
                TemporaryDownloadError,
                requests.RequestException,
            ) as error:
                coverage.loc[idx, "price_status"] = (
                    "coverage_only_" + type(error).__name__
                )
    finally:
        quotes.session.close()
    coverage["xbrl_status"] = (
        coverage["cik"].map(filings).fillna("no_cik_or_not_modeled")
    )
    # A synthetic CIK keeps price-only issuers in the panel. No fact can match it.
    synthetic = {ticker: -(i + 1) for i, ticker in enumerate(coverage["ticker"])}
    cohorts["cik"] = (
        cohorts["cik"].fillna(cohorts["ticker"].map(synthetic)).astype("int64")
    )
    facts = pd.concat(pieces, ignore_index=True)
    # Views hold only approved thematic data and use separate Week 4 output paths.
    with duckdb.connect() as connection:
        connection.register("universe", cohorts)
        connection.register("prices", prices)
        connection.register("company_facts", facts)
        connection.register("fred", fred)
        panel, _, market = build_monthly_panel(connection, settings2, reports)
    conditions = pd.read_parquet(directory / "macro_conditions.parquet")
    processed, quality = preprocess_panel(panel, FIRM_FEATURES, conditions)
    labels = make_labels(cohorts, prices, market, end=settings2["development_end"])
    meta, matrix = build_benchmark_sample(panel, processed, labels)
    if meta.empty:
        raise ValueError(
            "No thematic stock-month has a valid forecast price and target."
        )
    panel = panel.merge(
        cohorts[[*KEYS, "history_sessions"]], on=KEYS, validate="one_to_one"
    )
    panel["fundamental_coverage"] = panel[FUNDAMENTALS].notna().mean(axis=1)
    cover_summary = (
        panel.groupby("ticker")
        .agg(
            forecast_rows=("date", "size"),
            price_rows=("price_at_forecast", "sum"),
            fundamental_coverage=("fundamental_coverage", "mean"),
        )
        .join(meta.groupby("ticker").size().rename("scored_rows"))
    )
    coverage = coverage.merge(cover_summary.reset_index(), on="ticker", how="left")
    coverage.to_csv(directory / "universe_coverage.csv", index=False)
    quality.to_csv(directory / "feature_quality.csv", index=False)
    labels.to_csv(directory / "label_availability.csv", index=False)
    parquet(panel, directory / "features_raw.parquet")
    parquet(meta, directory / "meta.parquet")
    parquet(matrix, directory / "matrix.parquet")
    write_json(
        {
            "baseline_signature": gate["signature"],
            "candidates_sha256": file_sha256(Path(settings["candidates"])),
            "input_sha256": {
                name: file_sha256(directory / f"{name}.parquet")
                for name in ("features_raw", "meta", "matrix")
            },
            "short_history_sessions": settings["short_history_sessions"],
        },
        directory / "panel_receipt.json",
    )
    print(
        f"Thematic sample: {len(meta)} eligible labeled stock-months from {meta['ticker'].nunique()} names.",
        flush=True,
    )
