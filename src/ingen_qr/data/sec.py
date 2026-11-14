# mypy: disable-error-code=import-untyped

import json
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from ingen_qr.data.download import Downloader


def load_ticker_map(downloader: Downloader, refresh: bool = False) -> dict[str, int]:
    content = downloader.get(
        "https://www.sec.gov/files/company_tickers.json", refresh=refresh
    )
    directory = Path("data/raw/sec")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "company_tickers.json").write_bytes(content)
    return {
        str(row["ticker"]).upper(): int(row["cik_str"])
        for row in json.loads(content).values()
    }


def normalize_company_facts(payload: dict[str, Any], tags: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for namespace, concepts in payload["facts"].items():
        for tag, concept in concepts.items():
            if tag not in tags:
                continue
            for unit, observations in concept["units"].items():
                for item in observations:
                    rows.append(
                        {
                            "cik": int(payload["cik"]),
                            "namespace": namespace,
                            "tag": tag,
                            "unit": unit,
                            "value": float(item["val"]),
                            "start": item.get("start"),
                            "end": item["end"],
                            "filed": item["filed"],
                            "accn": item["accn"],
                            "form": item.get("form"),
                            "fy": item.get("fy"),
                            "fp": item.get("fp"),
                            "frame": item.get("frame"),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for name in ("start", "end", "filed"):
        frame[name] = pd.to_datetime(frame[name], format="%Y-%m-%d")
    calendar = xcals.get_calendar(
        "XNYS",
        start=frame["filed"].min() - pd.Timedelta(days=1),
        end=frame["filed"].max() + pd.Timedelta(days=14),
    )
    opens = {
        day: calendar.session_open(
            calendar.date_to_session(day + pd.Timedelta(days=1), direction="next")
        )
        for day in frame["filed"].unique()
    }
    frame["available_at"] = pd.to_datetime(frame["filed"].map(opens), utc=True)
    frame["date"] = frame["end"]
    frame["is_amendment"] = frame["form"].str.endswith("/A", na=False)
    frame["availability_rule"] = "next_session_open_after_filed_date"
    return frame.drop_duplicates().reset_index(drop=True)


def load_company_facts(
    cik: int,
    tags: list[str],
    downloader: Downloader,
    refresh: bool = False,
) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    content = downloader.get(url, refresh=refresh)
    directory = Path("data/raw/sec/companyfacts")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"CIK{cik:010d}.json").write_bytes(content)
    return normalize_company_facts(json.loads(content), tags)
