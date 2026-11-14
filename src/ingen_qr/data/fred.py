import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ingen_qr.data.download import Downloader

URL = "https://api.stlouisfed.org/fred/series/observations"
VINTAGE_URL = "https://api.stlouisfed.org/fred/series/vintagedates"
# FRED allows at most 2000 vintage dates per JSON request; keep a safety margin.
MAX_VINTAGES = 1900


def normalize_fred(records: list[dict[str, Any]], series_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No FRED observations for {series_id}.")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame["value"] = pd.to_numeric(frame["value"].replace(".", float("nan")))
    vintage = pd.to_datetime(frame["realtime_start"], format="%Y-%m-%d")
    known_day = pd.concat([frame["date"], vintage], axis=1).max(axis=1)
    frame["available_at"] = (
        (known_day + pd.Timedelta(days=1))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    frame["series_id"] = series_id
    frame["value_missing"] = frame["value"].isna()
    frame["availability_rule"] = "ALFRED_vintage_day_plus_one_day"
    return frame.drop_duplicates().reset_index(drop=True)


def _request(
    url: str,
    params: dict[str, str],
    series_id: str,
    downloader: Downloader,
    refresh: bool,
) -> bytes:
    try:
        return downloader.get(url, params=params, refresh=refresh)
    except requests.RequestException as error:
        raise RuntimeError(
            f"FRED {series_id}: {type(error).__name__}. Check the API key/network."
        ) from None


def _vintage_dates(
    series_id: str, api_key: str, downloader: Downloader, refresh: bool
) -> list[str]:
    dates: list[str] = []
    while True:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": "10000",
            "offset": str(len(dates)),
        }
        payload = json.loads(
            _request(VINTAGE_URL, params, series_id, downloader, refresh)
        )
        batch = payload["vintage_dates"]
        dates.extend(batch)
        if not batch or len(dates) >= int(payload["count"]):
            return dates


def load_fred(
    series_id: str,
    start: str,
    end: str,
    api_key: str,
    downloader: Downloader,
    refresh: bool = False,
) -> pd.DataFrame:
    directory = Path("data/raw/fred") / series_id
    directory.mkdir(parents=True, exist_ok=True)
    vintages = _vintage_dates(series_id, api_key, downloader, refresh)
    if not vintages:
        raise ValueError(f"FRED returned no vintage dates for {series_id}.")
    records: list[dict[str, Any]] = []
    for window, first in enumerate(range(0, len(vintages), MAX_VINTAGES)):
        chunk = vintages[first : first + MAX_VINTAGES]
        is_last = first + MAX_VINTAGES >= len(vintages)
        offset = 0
        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
                "realtime_start": "1776-07-04" if first == 0 else chunk[0],
                "realtime_end": "9999-12-31" if is_last else chunk[-1],
                "output_type": "1",
                "limit": "100000",
                "offset": str(offset),
            }
            content = _request(URL, params, series_id, downloader, refresh)
            payload = json.loads(content)
            batch = payload["observations"]
            name = f"window_{window:02d}_page_{offset:06d}.json"
            (directory / name).write_bytes(content)
            records.extend(batch)
            offset += len(batch)
            if not batch or offset >= int(payload["count"]):
                break
    return normalize_fred(records, series_id)
