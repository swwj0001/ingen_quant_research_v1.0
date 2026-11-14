# mypy: disable-error-code=import-untyped

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from yfinance.exceptions import YFTickerMissingError

from ingen_qr.data.download import Downloader, TemporaryDownloadError
from ingen_qr.data.factors import load_factors
from ingen_qr.data.fred import load_fred
from ingen_qr.data.membership import audit_membership, build_cohorts
from ingen_qr.data.price_panel import load_prices
from ingen_qr.data.sec import load_company_facts, load_ticker_map
from ingen_qr.data.storage import ParquetStore
from ingen_qr.data.universe import load_constituent_source


def credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not api_key or "YOUR_" in api_key or "PUT_" in api_key:
        raise ValueError("Fill FRED_API_KEY in .env first.")
    if not re.search(r"\S+@\S+\.\S+", agent) or "example.com" in agent:
        raise ValueError("SEC_USER_AGENT must contain your real contact email.")
    return api_key, agent


def build(config: dict[str, Any], smoke: bool, refresh: bool) -> None:
    api_key, agent = credentials()
    reports = Path("data/reports")
    reports.mkdir(parents=True, exist_ok=True)
    store = ParquetStore(Path("data/processed"))
    store.reset()
    market = Downloader(min_interval=2.0)
    public = Downloader(min_interval=0.5)
    sec = Downloader(min_interval=0.5, headers={"User-Agent": agent})
    start = str(config["study_start"])
    end = str(config["study_end"])
    market_proxy = str(config["market_proxy"])

    print("[1/5] Historical membership", flush=True)
    history = load_constituent_source(refresh=refresh)
    audit_membership(history).to_csv(reports / "membership_anchors.csv", index=False)
    cohorts = build_cohorts(history, start, end, str(config["universe_known_through"]))
    cohorts = cohorts.loc[cohorts["ticker"] != market_proxy].copy()
    if smoke:
        cohorts = cohorts.loc[cohorts["ticker"].isin(config["smoke_tickers"])].copy()
    tickers = sorted(cohorts["ticker"].unique())
    if not tickers:
        raise ValueError("No stocks in the selected universe.")
    aliases: dict[str, str] = config["ticker_aliases"]
    origin_dates = (
        cohorts["forecast_at"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    )
    required_days = set(origin_dates) | set(cohorts["date"].dt.strftime("%Y-%m-%d"))
    print("[2/5] Fama-French five factors and momentum", flush=True)
    factors = load_factors(public, refresh=refresh)
    factors = factors.loc[factors["date"].between("2008-01-01", end)]
    store.write(factors, "factors", "french")

    for series in config["fred_series"]:
        print(f"[3/5] FRED/ALFRED {series}", flush=True)
        frame = load_fred(series, "2008-01-01", end, api_key, public, refresh)
        store.write(frame, "fred", series)

    print("[4/5] SEC ticker-to-CIK reference", flush=True)
    ticker_map = load_ticker_map(sec, refresh=refresh)
    overrides = {name: int(cik) for name, cik in config["cik_overrides"].items()}
    cik_by_ticker: dict[str, int] = {}
    sec_report: list[dict[str, Any]] = []
    for ticker in tickers:
        provider = aliases.get(ticker, ticker)
        cik = overrides.get(ticker, ticker_map.get(provider))
        if cik is None:
            sec_report.append({"ticker": ticker, "status": "missing_CIK_mapping"})
        else:
            cik_by_ticker[ticker] = cik
    ciks = sorted(set(cik_by_ticker.values()))
    for position, cik in enumerate(ciks, 1):
        print(f"[4/5] Company facts {position}/{len(ciks)} CIK {cik}", flush=True)
        try:
            frame = load_company_facts(cik, config["sec_tags"], sec, refresh)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            sec_report.append({"cik": cik, "status": "companyfacts_not_found"})
            continue
        if frame.empty:
            sec_report.append({"cik": cik, "status": "no_selected_tags"})
            continue
        frame = frame.loc[frame["date"].between("2008-01-01", end)].copy()
        store.write(frame, "company_facts", str(cik))
        sec_report.append({"cik": cik, "status": "ok", "rows": len(frame)})
    pd.DataFrame(sec_report).to_csv(reports / "sec_coverage.csv", index=False)

    covered_days: dict[str, set[str]] = {}
    price_report: list[dict[str, Any]] = []
    price_tickers = sorted(set(tickers) | {market_proxy})
    market_proxy_rows = 0
    for position, ticker in enumerate(price_tickers, 1):
        provider = aliases.get(ticker, ticker)
        print(f"[5/5] Prices {position}/{len(price_tickers)} {ticker}", flush=True)
        try:
            frame = load_prices(provider, market, refresh=refresh)
        except TemporaryDownloadError:
            raise RuntimeError(
                f"Yahoo access failed for {ticker}. Build incomplete; keep the cache "
                "and rerun after restoring access."
            ) from None
        except YFTickerMissingError as error:
            if ticker == market_proxy:
                raise RuntimeError(
                    f"Market proxy {market_proxy} has no prices."
                ) from error
            price_report.append(
                {"ticker": ticker, "provider": provider, "status": type(error).__name__}
            )
            continue
        except ValueError as error:
            if ticker == market_proxy:
                raise RuntimeError(
                    f"Invalid market proxy {market_proxy} prices."
                ) from error
            price_report.append(
                {"ticker": ticker, "provider": provider, "status": str(error)}
            )
            continue
        frame = frame.loc[frame["date"] <= pd.Timestamp(end)].copy()
        frame["ticker"] = ticker
        frame["provider_ticker"] = provider
        if ticker == market_proxy:
            if frame.empty:
                raise RuntimeError(
                    f"Market proxy {market_proxy} has no study-period rows."
                )
            store.write(frame, "prices", ticker)
            market_proxy_rows = len(frame)
            continue
        days = set(frame["date"].dt.strftime("%Y-%m-%d"))
        covered_days[ticker] = days & required_days
        store.write(frame, "prices", ticker)
        price_report.append(
            {"ticker": ticker, "provider": provider, "status": "ok", "rows": len(frame)}
        )
    pd.DataFrame(price_report).to_csv(reports / "price_coverage.csv", index=False)

    entry_dates = cohorts["date"].dt.strftime("%Y-%m-%d")
    cohorts["feature_price_covered"] = [
        day in covered_days.get(ticker, set())
        for ticker, day in zip(cohorts["ticker"], origin_dates, strict=True)
    ]
    cohorts["entry_price_covered"] = [
        day in covered_days.get(ticker, set())
        for ticker, day in zip(cohorts["ticker"], entry_dates, strict=True)
    ]
    cohorts["cik"] = cohorts["ticker"].map(cik_by_ticker).astype("Int64")
    cohorts["cik_mapped"] = cohorts["cik"].notna()
    cohorts["provider_ticker"] = cohorts["ticker"].map(
        lambda ticker: aliases.get(ticker, ticker)
    )
    store.write(cohorts, "universe", "cohorts")
    monthly = cohorts.groupby("date").agg(
        pool_size=("ticker", "size"),
        feature_price_covered=("feature_price_covered", "sum"),
        entry_price_covered=("entry_price_covered", "sum"),
        cik_mapped=("cik_mapped", "sum"),
    )
    monthly.to_csv(reports / "monthly_coverage.csv")
    store.database(Path("data/ingen_qr.duckdb"))
    summary = {
        "mode": "smoke" if smoke else "full",
        "status": "built_with_coverage_report",
        "requested_stocks": len(tickers),
        "stocks_with_prices": len(covered_days),
        "market_proxy": market_proxy,
        "market_proxy_rows": market_proxy_rows,
        "requested_CIKs": len(ciks),
        "tables": sorted(store.written),
        "cache_hits": market.cache_hits + public.cache_hits + sec.cache_hits,
        "database": "data/ingen_qr.duckdb",
        "coverage": "data/reports/monthly_coverage.csv",
        "universe_known_through": str(config["universe_known_through"]),
    }
    (reports / "build_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    for client in (market, public, sec):
        client.session.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text("utf-8"))
    build(config, args.smoke, args.refresh)


if __name__ == "__main__":
    main()
