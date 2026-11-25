"""Rebuild and verify the frozen Week 2 comparison sample."""

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ingen_qr.evaluation.benchmarks import build_benchmark_sample
from ingen_qr.evaluation.splitters import (
    Fold,
    PurgedWalkForward,
    fold_split_sha256,
    split_manifest_sha256,
)

BASELINE_RUN_ID = "d14c41498e7e43ff9494a227213ee326"
KEYS = ["date", "ticker"]
FOLD_COLUMNS = [
    "test_month",
    "training_rows",
    "validation_rows",
    "refit_rows",
    "test_rows",
    "latest_training_label_end",
]
INPUT_NAMES = ("features_raw", "features_processed", "labels", "macro_conditions")


@dataclass(frozen=True)
class Baseline:
    status: str
    start_time: int
    end_time: int
    hashes: dict[str, str]
    split_sha256: str | None


@dataclass(frozen=True)
class Week3Sample:
    meta: pd.DataFrame
    matrix: pd.DataFrame
    manifest: pd.DataFrame
    folds: list[Fold]
    hashes: dict[str, str]
    split_sha256: str


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_baseline(database: Path) -> Baseline:
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT status, start_time, end_time FROM runs WHERE run_uuid = ?",
            (BASELINE_RUN_ID,),
        ).fetchone()
        if row is None or row[0] != "FINISHED" or row[2] is None:
            raise ValueError("The frozen Week 2 baseline must be FINISHED.")
        params = dict(
            conn.execute(
                "SELECT key, value FROM params WHERE run_uuid = ?", (BASELINE_RUN_ID,)
            )
        )
        tag = conn.execute(
            "SELECT value FROM tags WHERE run_uuid = ? AND key = 'split_sha256'",
            (BASELINE_RUN_ID,),
        ).fetchone()
    return Baseline(row[0], row[1], row[2], params, None if tag is None else tag[0])


def verify_saved_predictions(
    expected: pd.DataFrame, path: Path, matrix: pd.DataFrame
) -> None:
    saved = pd.read_csv(path, parse_dates=["date", "label_end"])
    if set(saved["model"]) != {"ols3", "momentum"}:
        raise ValueError(
            "The baseline prediction table must contain OLS-3 and momentum."
        )
    columns = [*KEYS, "label_excess_return", "label_end"]
    expected = expected[columns].sort_values(KEYS).reset_index(drop=True)
    for model in ("ols3", "momentum"):
        actual = (
            saved.loc[saved["model"] == model, columns]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
        )
    momentum = matrix[[*KEYS, "momentum_252_21d"]].merge(
        expected[KEYS], on=KEYS, validate="one_to_one"
    )
    actual_momentum = saved.loc[saved["model"] == "momentum", [*KEYS, "prediction"]]
    pd.testing.assert_series_equal(
        actual_momentum.sort_values(KEYS)["prediction"].reset_index(drop=True),
        momentum.sort_values(KEYS)["momentum_252_21d"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-14,
    )


def load_verified_sample(config_path: Path, baseline: Baseline) -> Week3Sample:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = Path(config["output_directory"])
    files = {name: data / f"{name}.parquet" for name in INPUT_NAMES}
    hashes = {"config_sha256": file_sha256(config_path)}
    hashes.update({f"{name}_sha256": file_sha256(path) for name, path in files.items()})
    for name, digest in hashes.items():
        if baseline.hashes.get(name) != digest:
            raise ValueError(f"Frozen Week 2 input fingerprint differs: {name}")

    joined, matrix = build_benchmark_sample(
        pd.read_parquet(files["features_raw"]),
        pd.read_parquet(files["features_processed"]),
        pd.read_parquet(files["labels"]),
    )
    pd.testing.assert_frame_equal(joined[KEYS], matrix[KEYS])
    splitter = PurgedWalkForward(
        first_test=str(config["first_test_month"]),
        last_test=str(config["last_test_month"]),
        validation_months=12,
        embargo_months=1,
    )
    records = []
    folds = []
    scored = []
    minimum = int(config["minimum_train_rows"])
    for fold in splitter.split(joined):
        train = joined.iloc[fold.refit]
        records.append(
            {
                "test_month": fold.test_month,
                "training_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "refit_rows": len(fold.refit),
                "test_rows": len(fold.test),
                "latest_training_label_end": (
                    train["label_end"].max() if len(train) else pd.NaT
                ),
                "split_sha256": fold_split_sha256(joined, fold),
            }
        )
        if len(train) >= minimum:
            folds.append(fold)
            scored.append(joined.iloc[fold.test])

    if not folds:
        raise ValueError("No eligible Week 2 baseline folds.")
    manifest = pd.DataFrame(records)
    reports = data / "reports"
    saved = pd.read_csv(
        reports / "walk_forward_folds.csv",
        parse_dates=["test_month", "latest_training_label_end"],
    )
    pd.testing.assert_frame_equal(
        manifest[FOLD_COLUMNS], saved[FOLD_COLUMNS], check_dtype=False
    )
    verify_saved_predictions(
        pd.concat(scored), reports / "benchmark_predictions.csv", matrix
    )
    return Week3Sample(
        joined,
        matrix.drop(columns=KEYS),
        manifest,
        folds,
        hashes,
        split_manifest_sha256(manifest),
    )
