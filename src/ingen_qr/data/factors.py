import csv
import json
from base64 import b64decode, b64encode
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ingen_qr.data.download import Downloader

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}
FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "ff_rf"]


def _parse_daily_zip(content: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("utf-8-sig")

    rows = list(csv.reader(StringIO(text)))
    header = next(row for row in rows if len(row) > 1 and not row[0].strip())
    names = [name.strip().lower().replace("-", "_") for name in header]
    names[0] = "date"

    records = [
        row
        for row in rows
        if row and len(row[0].strip()) == 8 and row[0].strip().isdigit()
    ]
    if not records:
        raise ValueError("No daily factor records found.")

    frame = pd.DataFrame(records, columns=names)
    frame["date"] = pd.to_datetime(frame["date"].str.strip(), format="%Y%m%d")

    for name in names[1:]:
        values = pd.to_numeric(frame[name])
        frame[name] = values.mask(values.isin([-99.99, -999.0])) / 100.0

    return frame.rename(columns={"rf": "ff_rf"})


def _load_factor_file(
    name: str,
    downloader: Downloader,
    refresh: bool,
) -> pd.DataFrame:
    url = BASE_URL + FILES[name]

    def fetch() -> bytes:
        with downloader.session.get(url, timeout=(10, 30)) as response:
            response.raise_for_status()
            content = response.content

        _parse_daily_zip(content)
        snapshot = {
            "source_url": url,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": sha256(content).hexdigest(),
            "content": b64encode(content).decode("ascii"),
        }
        return json.dumps(snapshot).encode("utf-8")

    packed = downloader.cached("french-source-v1:" + url, fetch, refresh=refresh)
    snapshot = json.loads(packed)
    content = b64decode(snapshot.pop("content"), validate=True)

    if sha256(content).hexdigest() != snapshot["sha256"]:
        raise ValueError("Factor file checksum mismatch.")

    directory = Path("data/raw/factors")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.zip").write_bytes(content)
    (directory / f"{name}.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )

    frame = _parse_daily_zip(content)
    frame["available_at"] = pd.Timestamp(snapshot["retrieved_at"])
    return frame


def load_factors(
    downloader: Downloader,
    refresh: bool = False,
) -> pd.DataFrame:
    ff5 = _load_factor_file("ff5_daily", downloader, refresh)
    momentum = _load_factor_file("momentum_daily", downloader, refresh)

    panel = ff5.merge(
        momentum,
        on="date",
        how="left",
        suffixes=("_ff", "_mom"),
        validate="one_to_one",
    )
    panel["available_at"] = panel[["available_at_ff", "available_at_mom"]].max(axis=1)
    panel = panel.drop(columns=["available_at_ff", "available_at_mom"])
    panel["factor_missing"] = panel[FACTORS].isna().any(axis=1)
    panel.attrs = {
        "usage": "ex_post_attribution_only",
        "availability_rule": "current_downloaded_snapshot",
    }
    return panel


if __name__ == "__main__":
    client = Downloader(min_interval=1.0)
    factors = load_factors(client)

    sample = factors.loc[
        factors["date"].between("2010-01-01", "2010-01-08"),
        ["date", *FACTORS],
    ]
    print(sample.head(3).to_string(index=False))
    print("First date:", factors["date"].min().date())
    print("Last date:", factors["date"].max().date())
    print("Missing factor rows:", factors["factor_missing"].sum())
    print("Available at:", factors["available_at"].max())
