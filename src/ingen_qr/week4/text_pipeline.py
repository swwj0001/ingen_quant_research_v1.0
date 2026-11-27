# mypy: disable-error-code=import-untyped

"""Build an auditable pilot or complete development-period filing corpus."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from ingen_qr.evaluation.week3_samples import file_sha256
from ingen_qr.week4 import baseline
from ingen_qr.week4.common import KEYS, WEEK2, config, parquet, read_json, write_json
from ingen_qr.week4.filings import (
    RateLimiter,
    audit_timezone,
    download_workers,
    filings_for,
    retrieve,
    sec_client,
    sec_rate,
)
from ingen_qr.week4.sentiment import Scorers, dictionary, nlp_runtime
from ingen_qr.week4.text_features import (
    EMBEDDING_LEVELS,
    EVENT_FEATURES,
    FEATURES,
    FILING_TIMESTAMPS,
    asof_features,
    supervised,
)

DOCUMENT_COLUMNS = [
    "cik",
    "accession",
    "form",
    "filing_date",
    "report_date",
    "items",
    "index_url",
    "document_url",
    "text",
    "sections",
    "text_sha256",
    "index_clock",
    "header_clock",
    "submissions_clock",
    "accepted_local",
    "accepted_utc",
    "available_at",
    "clock_source",
    "submissions_check",
    "status",
]
SCORE_COLUMNS = [
    "lm_tone",
    "lm_uncertainty",
    "lm_litigious",
    "finbert_tone",
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "embedding",
    "document_sentences",
    "nlp_sampled_sentences",
    "lm_seconds",
    "finbert_seconds",
    "embedding_seconds",
]


def select_pilot(rows: pd.DataFrame, start: str, end: str, count: int) -> pd.DataFrame:
    sample = rows.loc[rows["date"].between(start, end)]
    coverage = (
        sample.assign(month=sample["date"].dt.to_period("M"))
        .groupby("cik")["month"]
        .nunique()
    )
    needed = sample["date"].dt.to_period("M").nunique()
    selected = coverage.loc[coverage.eq(needed)].sort_index().index[:count]
    if len(selected) != count or needed != 36:
        raise ValueError(
            f"Expected {count} continuously represented issuers across 36 months."
        )
    return sample.loc[sample["cik"].isin(selected)].copy()


def deduplicate(records: pd.DataFrame) -> pd.DataFrame:
    records = records.sort_values(["accepted_utc", "accession"], na_position="last")
    usable = records["status"].eq("ok") & records["report_date"].ne("")
    duplicates = records.loc[usable].duplicated(
        ["cik", "form", "report_date", "items", "text_sha256"]
    )
    records.loc[duplicates.index[duplicates], "status"] = "duplicate_narrative"
    return records


def coverage(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in features.groupby("date"):
        for name in FEATURES:
            values = group[name]
            observed = values.notna() if name not in EVENT_FEATURES else values.gt(0)
            rows.append(
                {
                    "date": date,
                    "feature": name,
                    "rows": len(group),
                    "observed": int(observed.sum()),
                    "coverage": float(observed.mean()),
                }
            )
    return pd.DataFrame(rows)


def download_issuer(
    cik: int,
    settings: dict[str, Any],
    cache_dir: str,
    limiter: RateLimiter,
    calendar: Any,
    start: str,
    end: str,
    doc_path: Path,
    timing_path: Path,
) -> None:
    """Download and parse one issuer in a worker thread with its own HTTP session."""
    client = sec_client(settings, cache_dir, limiter)
    try:
        # Filing date can roll to the next business day; acceptance time sets the final filter.
        discovery_start = (
            (pd.Timestamp(start) - pd.Timedelta(days=7)).date().isoformat()
        )
        discovery_end = min(
            "2023-12-31",
            (pd.Timestamp(end) + pd.Timedelta(days=7)).date().isoformat(),
        )
        records = [
            retrieve(
                filing,
                client,
                calendar,
                int(settings["minimum_section_words"]),
                start,
                end,
            )
            for filing in filings_for(client, cik, discovery_start, discovery_end)
        ]
        corpus = deduplicate(pd.DataFrame(records, columns=DOCUMENT_COLUMNS))
        calls, misses, seconds = client.measurements()
        parquet(corpus, doc_path)
        # The timing file is written last and marks the issuer download as complete.
        write_json(
            {
                "sec_calls": calls,
                "sec_uncached_calls": misses,
                "sec_http_requests": client.http_requests,
                "sec_download_seconds": seconds,
            },
            timing_path,
        )
    finally:
        client.session.close()


def earlier_download_timing(
    cache_root: Path, issuers: list[int], current: Path
) -> tuple[int, float, int]:
    """First-fetch network timings recorded by earlier runs for these issuers.

    Returns uncached download calls, their network seconds and the HTTP requests
    they sent. Records written before HTTP requests were counted assume one
    request per uncached call.
    """
    misses, seconds, http = 0, 0.0, 0
    for cik in issuers:
        for path in sorted(cache_root.glob(f"*/{cik}_download.json")):
            if path.parent == current:
                continue
            record = read_json(path)
            uncached = int(record.get("sec_uncached_calls", 0))
            if uncached > 0:
                misses += uncached
                seconds += float(record["sec_download_seconds"])
                http += int(record.get("sec_http_requests", uncached))
                break
    return misses, seconds, http


def estimate_runtime(
    issuers: pd.DataFrame,
    documents: pd.DataFrame,
    pilot_years: int,
    full_years: int,
    clock_sample: tuple[int, int, float],
    rate: float,
    workers: int,
    earlier: tuple[int, float, int] = (0, 0.0, 0),
) -> pd.DataFrame:
    factor = full_years / pilot_years
    calls = int(issuers["sec_calls"].sum())
    misses = int(issuers["sec_uncached_calls"].sum())
    network = float(issuers["sec_download_seconds"].sum())
    http = (
        int(issuers["sec_http_requests"].sum())
        if "sec_http_requests" in issuers
        else misses
    )
    if misses:
        seconds_per_call = network / misses
        requests_per_call = http / misses
        basis = "pilot uncached SEC requests"
    elif earlier[0]:
        seconds_per_call = earlier[1] / earlier[0]
        requests_per_call = earlier[2] / earlier[0]
        basis = "first-download SEC timings from an earlier pilot run"
    else:
        _, samples, seconds = clock_sample
        seconds_per_call = seconds / samples
        requests_per_call = 1.0
        basis = "uncached SEC clock check (pilot corpus was cached)"
    # Each download call may send several HTTP requests (redirects, retries); the
    # shared limiter caps HTTP requests, while concurrent workers overlap network time.
    requests_per_call = max(requests_per_call, 1.0)
    per_call = max(requests_per_call / rate, seconds_per_call / workers)
    basis += (
        f"; {requests_per_call:.2f} HTTP requests per download call; "
        f"{workers} workers, at most {rate:g} requests/s"
    )
    estimates = [
        ("SEC download", network, calls * per_call, basis),
        (
            "FinBERT",
            documents["finbert_seconds"].sum(),
            documents["finbert_seconds"].sum(),
            "pilot scoring",
        ),
        (
            "Sentence embeddings",
            documents["embedding_seconds"].sum(),
            documents["embedding_seconds"].sum(),
            "pilot scoring",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "pilot_measured_seconds": measured,
                "pilot_issuer_years": pilot_years,
                "full_issuer_years": full_years,
                "estimated_full_hours": full_seconds * factor / 3600,
                "basis": method,
            }
            for stage, measured, full_seconds, method in estimates
        ]
    )


def run(settings: dict[str, Any], root: Path, gate: dict[str, Any], mode: str) -> None:
    is_pilot = mode == "pilot"
    pilot_receipt = root / "pilot" / "completed.json"
    if str(settings["development_end"]) != "2023-12-31":
        raise ValueError(
            "Week 4 development cutoff must be 2023-12-31. Holdout is sealed."
        )
    lexicon_path = Path(settings["dictionary"])
    lexicon = dictionary(lexicon_path)
    data = baseline.sample()
    frozen = config(WEEK2)
    raw = Path(frozen["output_directory"]) / "features_raw.parquet"
    mapping = pd.read_parquet(raw, columns=[*KEYS, "cik", "forecast_at"])
    if is_pilot:
        rows = mapping.merge(
            data.meta[
                [*KEYS, "label_available_at", "label_end", "label_excess_return"]
            ],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    else:
        rows = data.meta.merge(
            mapping[[*KEYS, "cik"]], on=KEYS, how="left", validate="one_to_one"
        )
    if rows["cik"].isna().any():
        raise ValueError(
            "Could not map all frozen research rows to their historical CIK."
        )
    rows["cik"] = rows["cik"].astype(int)
    rows["forecast_at"] = pd.to_datetime(rows["forecast_at"], utc=True)
    start = str(settings["pilot_start"] if is_pilot else settings["filing_start"])
    end = str(settings["pilot_end"] if is_pilot else settings["development_end"])
    if start > end or end > "2023-12-31":
        raise ValueError("Invalid development-only filing dates.")
    if is_pilot:
        rows = select_pilot(rows, start, end, int(settings["pilot_issuer_count"]))
    issuers = rows["cik"].unique()
    base = root / mode
    base.mkdir(parents=True, exist_ok=True)
    calendar = xcals.get_calendar("XNYS", start="2009-12-01", end="2024-02-01")
    receipt = base / "completed.json"
    # Resolve 'auto' before the cache key so CPU and GPU scores never share a cache.
    device, device_name, batch = nlp_runtime(settings)
    print(f"NLP device: {device} ({device_name}), batch size {batch}.", flush=True)
    scorer_settings = {**settings, "nlp_device": device, "nlp_batch_size": batch}
    identity = {
        **settings,
        "nlp_device_resolved": device,
        "nlp_batch_size_resolved": batch,
        "dictionary_sha256": file_sha256(lexicon_path),
        "code": {
            p.name: file_sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("filings.py"),
                Path(__file__).with_name("sentiment.py"),
                Path(__file__).with_name("text_features.py"),
            ]
        },
    }
    if not is_pilot:
        pilot = read_json(pilot_receipt) if pilot_receipt.is_file() else {}
        if (
            pilot.get("baseline_signature") != gate["signature"]
            or pilot.get("text_config") != identity
        ):
            raise RuntimeError(
                "Finish the 20 × 3-year pilot with the current text settings before building the full corpus."
            )
    receipt.unlink(missing_ok=True)
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    scored_path = base / "issuer_cache" / signature
    scored_path.mkdir(parents=True, exist_ok=True)
    extraction_start = perf_counter()
    scored: list[pd.DataFrame] = []
    audit: list[pd.DataFrame] = []
    issuer_rows: list[dict[str, Any]] = []
    scorer: Scorers | None = None

    cache_dir = str(root / "cache" / "sec")
    rate, workers = sec_rate(settings), download_workers(settings)
    limiter = RateLimiter(rate)
    client = sec_client(settings, cache_dir, limiter)
    try:
        pd.DataFrame(audit_timezone(client, calendar, refresh=is_pilot)).to_csv(
            base / "sec_timezone_sample.csv", index=False
        )
        clock_sample = client.measurements()
    finally:
        client.session.close()
    ordered = [int(cik) for cik in sorted(issuers)]
    paths = {
        cik: (
            scored_path / f"{cik}_filings.parquet",
            scored_path / f"{cik}_scores.parquet",
            scored_path / f"{cik}_download.json",
        )
        for cik in ordered
    }
    pending = [
        cik
        for cik in ordered
        if not (paths[cik][0].is_file() and paths[cik][2].is_file())
    ]
    print(
        f"{mode}: {len(issuers)} issuers, {start} through {end}. "
        f"{len(pending)} to download with {workers} workers at up to {rate:g} SEC requests/s.",
        flush=True,
    )
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sec")
    futures = {
        cik: executor.submit(
            download_issuer,
            cik,
            settings,
            cache_dir,
            limiter,
            calendar,
            start,
            end,
            paths[cik][0],
            paths[cik][2],
        )
        for cik in pending
    }
    try:
        for i, cik in enumerate(ordered, 1):
            if i == 1 or i % 10 == 0:
                print(
                    f"{mode} SEC coverage {i}/{len(issuers)} (CIK {cik:010d})",
                    flush=True,
                )
            doc_path, score_path, timing_path = paths[cik]
            elapsed = perf_counter()
            if cik in futures:
                # Downloads run ahead in worker threads; scoring stays in issuer order.
                futures[cik].result()
            cached = cik not in futures and score_path.is_file()
            corpus = pd.read_parquet(doc_path)
            download = read_json(timing_path)
            if cached:
                scores = pd.read_parquet(score_path)
            else:
                accepted = corpus.loc[corpus["status"].eq("ok")]
                records = []
                memo: dict[str, dict[str, Any]] = {}
                for row in accepted.to_dict("records"):
                    if scorer is None:
                        scorer = Scorers(scorer_settings, lexicon)
                    digest = row["text_sha256"]
                    if digest in memo:
                        values = {
                            **memo[digest],
                            "lm_seconds": 0.0,
                            "finbert_seconds": 0.0,
                            "embedding_seconds": 0.0,
                        }
                    else:
                        values = scorer.score(row["text"])
                        memo[digest] = values
                    records.append(
                        {
                            **{
                                k: row[k]
                                for k in DOCUMENT_COLUMNS
                                if k not in {"text", "sections"}
                            },
                            **values,
                        }
                    )
                scores = pd.DataFrame(
                    records,
                    columns=[
                        *[k for k in DOCUMENT_COLUMNS if k not in {"text", "sections"}],
                        *SCORE_COLUMNS,
                    ],
                )
                parquet(scores, score_path)
            audit.append(corpus.drop(columns=["text", "sections"]))
            if not scores.empty:
                scored.append(scores)
            issuer_rows.append(
                {
                    "cik": cik,
                    "candidate_filings": len(corpus),
                    "scored_filings": len(scores),
                    "cache_reused": cached,
                    **download,
                    "wall_seconds": perf_counter() - elapsed,
                }
            )
    except BaseException:
        print(
            "Stopping: waiting for running downloads to finish their current issuer.",
            flush=True,
        )
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    audit_frame = pd.concat(audit, ignore_index=True)
    audit_frame.to_csv(base / "filing_coverage.csv", index=False)
    audit_frame.loc[
        audit_frame["submissions_check"].eq("mismatch")
        | audit_frame["status"].isin(
            ["unconfirmed_acceptance", "authoritative_clock_conflict"]
        )
    ].to_csv(base / "acceptance_exceptions.csv", index=False)
    issuer_coverage = pd.DataFrame(issuer_rows)
    issuer_coverage.to_csv(base / "issuer_coverage.csv", index=False)
    if not scored:
        raise RuntimeError(
            "No usable EDGAR narrative was found. Inspect filing_coverage.csv."
        )
    documents = pd.concat(scored, ignore_index=True)
    if documents["accession"].duplicated().any():
        # The same multi-filer accession can appear under multiple CIKs.
        documents["accession"] = (
            documents["cik"].astype(str) + ":" + documents["accession"]
        )
    for col in ["accepted_utc", "available_at"]:
        documents[col] = pd.to_datetime(documents[col], utc=True)
    parquet(documents, base / "filing_scores.parquet")
    features = asof_features(rows, documents, int(settings["max_filing_age_days"]))
    features, head = supervised(features, rows, documents, settings)
    if (
        features[FILING_TIMESTAMPS]
        .gt(features["forecast_at"], axis="index")
        .any()
        .any()
    ):
        raise ValueError("A text feature used future filing content.")
    if not features[EMBEDDING_LEVELS].notna().any().any():
        raise RuntimeError(
            "No method (c) scores were produced. Inspect text coverage and minimum mature rows."
        )
    path = base / "firm_month_features.parquet"
    parquet(features, path)
    head.to_csv(base / "head_training_audit.csv", index=False)
    coverage(features).to_csv(base / "firm_month_coverage.csv", index=False)
    if is_pilot:
        pilot_years = len(issuers) * (
            pd.Timestamp(end).year - pd.Timestamp(start).year + 1
        )
        full_issuers = (
            data.meta[KEYS]
            .merge(mapping[[*KEYS, "cik"]], on=KEYS, validate="one_to_one")["cik"]
            .nunique()
        )
        full_years = full_issuers * (
            pd.Timestamp(str(settings["development_end"])).year
            - pd.Timestamp(str(settings["filing_start"])).year
            + 1
        )
        forecast = estimate_runtime(
            issuer_coverage,
            documents,
            pilot_years,
            full_years,
            clock_sample,
            rate,
            workers,
            earlier_download_timing(scored_path.parent, ordered, scored_path),
        )
        forecast.to_csv(base / "full_runtime_estimate.csv", index=False)
        print(
            f"Full corpus estimate ({pilot_years:,} pilot to {full_years:,} planned issuer-years):",
            flush=True,
        )
        for item in forecast.itertuples(index=False):
            print(
                f"  {item.stage}: {item.estimated_full_hours:.2f} h ({item.basis})",
                flush=True,
            )
    cost = settings["compute_cost_usd_per_hour"]
    runtime = []
    for method, seconds, setup in [
        ("Loughran–McDonald", documents["lm_seconds"].sum(), 0.0),
        (
            "FinBERT",
            documents["finbert_seconds"].sum(),
            scorer.finbert_setup_seconds if scorer else 0.0,
        ),
        (
            "Sentence embeddings and supervised ridge",
            documents["embedding_seconds"].sum() + head["seconds"].sum(),
            scorer.embedding_setup_seconds if scorer else 0.0,
        ),
    ]:
        runtime.append(
            {
                "method": method,
                "scoring_seconds": seconds,
                "setup_seconds_this_run": setup,
                "device": device,
                "scored_filings": len(documents),
                "estimated_compute_cost_usd": (
                    None if cost is None else float(cost) * (seconds + setup) / 3600
                ),
            }
        )
    pd.DataFrame(runtime).to_csv(base / "method_runtime.csv", index=False)
    write_json(
        {
            "mode": mode,
            "baseline_signature": gate["signature"],
            "features_sha256": file_sha256(path),
            "text_config": identity,
            "issuer_cache": scored_path.as_posix(),
            "issuers": len(issuers),
            "rows": len(rows),
            "filings": len(documents),
            "filing_statuses": dict(Counter(audit_frame["status"])),
            "wall_seconds_this_run": perf_counter() - extraction_start,
            "nlp_device_used": device,
            "nlp_batch_size_used": batch,
            "sec_download_workers": workers,
            "sec_max_requests_per_second": rate,
        },
        receipt,
    )
    print(
        f"{mode} complete: {len(issuers)} issuers, {len(documents)} scored filings, {len(rows)} firm-months.",
        flush=True,
    )
