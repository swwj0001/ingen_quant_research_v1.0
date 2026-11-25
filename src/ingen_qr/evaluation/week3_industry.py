"""Current SEC SIC snapshot used for Week 3's FF12 diagnostic."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from ingen_qr.data.download import Downloader


def classify_sic(sic: object, bins: pd.DataFrame) -> tuple[int | None, str]:
    text = str(sic).strip()
    if not text.isdecimal() or not 1 <= int(text) <= 9999:
        return None, "Other"
    number = int(text)
    match = bins.loc[bins["sic_start"].le(number) & bins["sic_end"].ge(number)]
    industry = str(match.iloc[0]["industry"]) if len(match) else "Other"
    return number, industry


def industry_snapshot(
    ciks: pd.Series, bins_path: Path, data: Path, agent: str = ""
) -> pd.DataFrame:
    path = data / "current_sec_sic.parquet"
    if path.exists():
        snapshot = pd.read_parquet(path)
        if set(ciks.dropna().astype(int)).issubset(set(snapshot["cik"])):
            return snapshot
        raise ValueError("The existing SIC snapshot does not cover this run's CIKs.")
    if not agent.strip():
        raise ValueError("Set SEC_USER_AGENT to a team name and contact email.")
    data.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(
        cache_dir=str(data / "sec_cache"),
        min_interval=0.5,
        headers={"User-Agent": agent},
    )
    bins = pd.read_csv(bins_path)
    records = []
    issuers = sorted(set(ciks.dropna().astype(int)))
    for index, cik in enumerate(issuers, start=1):
        if index == 1 or index % 50 == 0 or index == len(issuers):
            print(f"SEC SIC {index}/{len(issuers)}", flush=True)
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        try:
            payload = json.loads(downloader.get(url))
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            payload = {}
        sic, industry = classify_sic(payload.get("sic"), bins)
        records.append(
            {
                "cik": cik,
                "sic": sic,
                "industry": industry,
                "sic_available": sic is not None,
            }
        )
    snapshot = pd.DataFrame(
        records, columns=["cik", "sic", "industry", "sic_available"]
    ).assign(snapshot_at=datetime.now(UTC).isoformat())
    snapshot.to_parquet(path, index=False)
    return snapshot
