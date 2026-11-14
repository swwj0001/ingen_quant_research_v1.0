# mypy: disable-error-code=import-untyped

"""Development-only recovery with bounded requests and explicit failure records."""

from pathlib import Path

import pandas as pd
import requests
from yfinance.exceptions import YFTickerMissingError

from ingen_qr.data.download import Downloader, TemporaryDownloadError
from ingen_qr.data.price_panel import _prepare_price_panel
from ingen_qr.data.prices import load_price_source


def recover_development_prices(
    prices: pd.DataFrame,
    reports: Path,
    raw: Path,
    end: str,
    allow_network: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = reports / "price_coverage.csv"
    if not coverage.exists():
        return prices.copy(), pd.DataFrame(
            columns=["ticker", "provider", "original_status", "result", "rows"]
        )
    failures = pd.read_csv(coverage).loc[lambda x: x["status"] != "ok"]
    results, recovered = [], []
    downloader = Downloader(min_interval=2.0, max_attempts=2)
    upper = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    for row in failures.itertuples():
        result: dict[str, object] = {
            "ticker": row.ticker,
            "provider": row.provider,
            "original_status": row.status,
            "result": "unresolved",
            "rows": 0,
        }
        snapshots = sorted(
            raw.glob(str(row.provider) + "_*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        source: pd.DataFrame | None = None
        if snapshots:
            source = pd.read_parquet(snapshots[0])
            result["source_file"] = snapshots[0].as_posix()
        elif allow_network and row.status != "YFTzMissingError":
            try:
                source = load_price_source(
                    str(row.provider), "2008-01-01", upper, downloader
                )
                result["source_file"] = "bounded_yahoo_request"
            except (
                ValueError,
                YFTickerMissingError,
                requests.RequestException,
                TemporaryDownloadError,
            ) as error:
                result["result"] = type(error).__name__ + ": " + str(error)
        if source is not None:
            dates = pd.DatetimeIndex(source.index)
            if dates.tz is not None:
                dates = dates.tz_convert("America/New_York").tz_localize(None)
            inside = dates.normalize() <= pd.Timestamp(end)
            future_splits = source.loc[~inside, "Stock Splits"].replace(0.0, 1.0)
            factor = float(future_splits.to_numpy(dtype=float).prod())
            source = source.loc[inside].copy()
            for column in ["Open", "High", "Low", "Close", "Dividends"]:
                source[column] *= factor
            source["Volume"] /= factor
            if not source.empty:
                try:
                    panel = _prepare_price_panel(source, str(row.ticker))
                    panel["provider_ticker"] = str(row.provider)
                    recovered.append(panel)
                    result.update(result="recovered_development_rows", rows=len(panel))
                except (ValueError, KeyError) as error:
                    result["result"] = type(error).__name__ + ": " + str(error)
        results.append(result)
    downloader.session.close()
    result_prices = pd.concat([prices, *recovered], ignore_index=True)
    result_prices = result_prices.sort_values("available_at").drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    return result_prices, pd.DataFrame(results)
