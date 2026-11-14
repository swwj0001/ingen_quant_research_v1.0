import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from ingen_qr.data.download import Downloader

SOURCE_VERSION = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
SOURCE_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    f"{SOURCE_VERSION}/{quote(SOURCE_FILE, safe='')}"
)
EXPECTED_SHA256 = "36326709d46d6cd25834de5df457b16f5f96fad3a06b9beac28f7b88aa0b0d54"
RAW_FILE = Path("data/raw/universe/sp500_history.csv")


def load_constituent_source(refresh: bool = False) -> pd.DataFrame:
    content = Downloader().get(SOURCE_URL, refresh=refresh)
    checksum = sha256(content).hexdigest()

    if checksum != EXPECTED_SHA256:
        raise ValueError("Source file does not match the pinned version.")

    frame = pd.read_csv(BytesIO(content), dtype=str)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_bytes(content)

    metadata = {
        "source_url": SOURCE_URL,
        "commit": SOURCE_VERSION,
        "sha256": checksum,
        "snapshots": len(frame),
        "first_change": frame["date"].min().strftime("%Y-%m-%d"),
        "last_change": frame["date"].max().strftime("%Y-%m-%d"),
        "availability_status": "pending_announcement_checks",
    }
    RAW_FILE.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return frame


def constituents_on(history: pd.DataFrame, as_of: str) -> list[str]:
    target = pd.to_datetime(as_of, format="%Y-%m-%d")
    first = history["date"].min()
    last = history["date"].max()

    if target < first or target > last:
        raise ValueError("Requested date is outside the source date range.")

    selected = history.loc[history["date"] <= target]
    tickers: str = selected.iloc[-1]["tickers"]
    return tickers.split(",")


if __name__ == "__main__":
    data = load_constituent_source()
    print("Snapshots:", len(data))
    print("First change:", data["date"].min().date())
    print("Last change:", data["date"].max().date())
    print("Latest member count:", len(data.iloc[-1]["tickers"].split(",")))
    print("Saved:", RAW_FILE)
