# mypy: disable-error-code=import-untyped

from datetime import UTC, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd

from ingen_qr.data.download import Downloader
from ingen_qr.data.prices import load_price_source


def _prepare_price_panel(source: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = source.sort_index().copy()
    frame.index = (
        pd.DatetimeIndex(frame.index)
        .tz_convert("America/New_York")
        .tz_localize(None)
        .normalize()
    )

    columns = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    if frame[columns].isna().any().any() or (frame["Close"] <= 0).any():
        raise ValueError(f"Missing or invalid prices for {ticker}.")
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate trading dates for {ticker}.")

    splits = frame["Stock Splits"].replace(0.0, 1.0)
    future_splits = splits.iloc[::-1].cumprod().iloc[::-1] / splits

    panel = pd.DataFrame(index=frame.index)
    panel["ticker"] = ticker
    for name in ("Open", "High", "Low", "Close"):
        panel["raw_" + name.lower()] = frame[name] * future_splits

    dividends = frame["Dividends"] * future_splits
    growth = (panel["raw_close"] + dividends) * splits / panel["raw_close"].shift()
    growth.iloc[0] = 1.0
    total_return = growth.cumprod() * 100.0
    adjustment = total_return / panel["raw_close"]

    for name in ("Open", "High", "Low", "Close"):
        panel["adjusted_" + name.lower()] = panel["raw_" + name.lower()] * adjustment

    panel["volume"] = frame["Volume"] / future_splits
    panel["dividend"] = dividends
    panel["split_ratio"] = splits

    calendar = xcals.get_calendar(
        "XNYS", start=frame.index.min(), end=frame.index.max()
    )
    closes = calendar.schedule.loc[frame.index, "close"]
    panel["available_at"] = pd.DatetimeIndex(closes) + pd.Timedelta(minutes=30)

    retrieved_at = pd.Timestamp(source.attrs["retrieved_at"])
    panel = panel.loc[panel["available_at"] <= retrieved_at].copy()
    panel.attrs = {
        **source.attrs,
        "adjustment_method": "forward_total_return_base_100",
        "availability_rule": "XNYS_close_plus_30_minutes_assumption",
    }
    return panel.reset_index(names="date")


def load_prices(
    ticker: str,
    downloader: Downloader,
    refresh: bool = False,
) -> pd.DataFrame:
    end = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
    source = load_price_source(ticker, "2008-01-01", end, downloader, refresh=refresh)
    return _prepare_price_panel(source, ticker)


if __name__ == "__main__":
    prices = load_prices("AAPL", Downloader(min_interval=1.0))
    sample = prices.loc[
        prices["date"].between("2010-01-01", "2010-01-31"),
        ["date", "raw_close", "adjusted_close", "volume", "available_at"],
    ]
    print(sample.head(3).to_string(index=False))
    print("Missing available_at:", prices["available_at"].isna().sum())
