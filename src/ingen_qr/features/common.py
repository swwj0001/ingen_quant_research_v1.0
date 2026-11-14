# mypy: disable-error-code=import-untyped

"""Trading-session alignment and availability-aware trailing windows."""

from collections.abc import Iterator

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
NAT = np.iinfo(np.int64).min
KEYS = ["ticker", "date", "available_at"]


def prepare_prices(prices: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = [*KEYS, "adjusted_close", *extra]
    missing = set(columns) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("No prices supplied.")
    frame = prices[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].dt.tz is not None:
        raise ValueError("date must contain timezone-naive trading dates.")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("date must contain midnight trading dates.")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    if frame[KEYS].isna().any().any():
        raise ValueError("Missing ticker, date or available_at.")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Select one as-of record per ticker and date.")
    for column in ["adjusted_close", *extra]:
        observed = frame[column].dropna()
        if not np.isfinite(observed).all():
            raise ValueError(f"Non-finite observed {column}.")
        if column == "volume":
            if (observed < 0).any():
                raise ValueError("volume must be non-negative.")
        elif (observed <= 0).any():
            raise ValueError(f"{column} must be positive.")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def price_blocks(
    frame: pd.DataFrame,
) -> Iterator[tuple[str, pd.DataFrame, pd.DatetimeIndex]]:
    calendar = xcals.get_calendar(
        "XNYS", start=frame["date"].min(), end=frame["date"].max()
    )
    sessions = pd.DatetimeIndex(calendar.sessions).tz_localize(None)
    if not frame["date"].isin(sessions).all():
        raise ValueError("A price date is not an XNYS trading session.")
    for ticker in frame["ticker"].unique():
        group = frame.loc[frame["ticker"] == ticker]
        observed = pd.DatetimeIndex(group["date"])
        grid = sessions[(sessions >= observed.min()) & (sessions <= observed.max())]
        block = group.set_index("date").reindex(grid)
        yield str(ticker), block, observed


def timestamp_ns(values: pd.Series) -> IntArray:
    return np.asarray(pd.DatetimeIndex(values).as_unit("ns").asi8, dtype=np.int64)


def daily_returns(block: pd.DataFrame) -> tuple[FloatArray, IntArray]:
    price = block["adjusted_close"]
    values = (price / price.shift(1) - 1.0).to_numpy(dtype=float, na_value=np.nan)
    known = timestamp_ns(block["available_at"])
    previous = np.concatenate((np.array([NAT], dtype=np.int64), known[:-1]))
    return values, np.maximum(known, previous)


def window_values(
    values: FloatArray, known: IntArray, cutoffs: IntArray, width: int
) -> FloatArray:
    padded = np.pad(values, (width - 1, 0), constant_values=np.nan)
    times = np.pad(known, (width - 1, 0), constant_values=NAT)
    matrix = np.lib.stride_tricks.sliding_window_view(padded, width)
    times_matrix = np.lib.stride_tricks.sliding_window_view(times, width)
    valid = (
        np.isfinite(matrix)
        & (times_matrix != NAT)
        & (cutoffs[:, None] != NAT)
        & (times_matrix <= cutoffs[:, None])
    )
    return np.where(valid, matrix, np.nan)


def minimum_count(width: int) -> int:
    return (4 * width + 4) // 5


def window_mean(values: FloatArray, width: int) -> FloatArray:
    count = np.isfinite(values).sum(axis=1)
    result = np.nansum(values, axis=1) / np.maximum(count, 1)
    return np.where(count >= minimum_count(width), result, np.nan)


def window_std(values: FloatArray, width: int) -> FloatArray:
    count = np.isfinite(values).sum(axis=1)
    mean = np.nansum(values, axis=1) / np.maximum(count, 1)
    squares = np.where(np.isfinite(values), (values - mean[:, None]) ** 2, 0.0)
    result = np.sqrt(squares.sum(axis=1) / np.maximum(count - 1, 1))
    return np.where(count >= minimum_count(width), result, np.nan)


def output_rows(
    block: pd.DataFrame, observed: pd.DatetimeIndex, ticker: str, features: list[str]
) -> pd.DataFrame:
    result = block.loc[observed, ["available_at", *features]].copy()
    result["ticker"] = ticker
    result.index.name = "date"
    return result.reset_index()[[*KEYS, *features]]
