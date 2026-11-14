# mypy: disable-error-code=import-untyped

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException as CurlRequestError
from yfinance.exceptions import YFRateLimitError

from ingen_qr.data.download import Downloader, TemporaryDownloadError


def load_price_source(
    ticker: str,
    start: str,
    end: str,
    downloader: Downloader,
    refresh: bool = False,
) -> pd.DataFrame:
    def fetch() -> bytes:
        yf.set_tz_cache_location("data/cache/yfinance")
        yf.config.debug.hide_exceptions = False

        try:
            frame: pd.DataFrame = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                timeout=30,
            )
        except (YFRateLimitError, CurlRequestError) as error:
            raise TemporaryDownloadError(
                f"Temporary price download error for {ticker}."
            ) from error

        if frame.empty:
            raise ValueError(f"No daily prices returned for {ticker}.")

        frame.attrs = {
            "ticker": ticker,
            "source": "Yahoo Finance",
            "yfinance_version": yf.__version__,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        return frame.to_parquet()

    key = f"yahoo-daily-v1:{ticker}:{start}:{end}"
    content = downloader.cached(key, fetch, refresh=refresh)

    directory = Path("data/raw/prices")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticker}_{start}_{end}.parquet"
    path.write_bytes(content)
    return pd.read_parquet(BytesIO(content))


if __name__ == "__main__":
    client = Downloader(min_interval=1.0)
    prices = load_price_source("AAPL", "2010-01-01", "2010-02-01", client)
    cached = load_price_source("AAPL", "2010-01-01", "2010-02-01", client)

    print("Rows:", len(prices))
    print("Cache hits:", client.cache_hits)
    print("Same content:", prices.equals(cached))
    print(prices[["Close", "Adj Close", "Volume"]].head(3))
