"""Shared Week 4 paths, atomic outputs and local experiment tracking."""

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ingen_qr.evaluation.week3_samples import file_sha256

KEYS = ["date", "ticker"]
DATABASE = Path("mlruns/tracking.db")
TEXT_CONFIG = Path("configs/week4_text.yaml")
WEEK2 = Path("configs/week2.yaml")
WEEK3 = Path("configs/week3.yaml")
WEEK3_REPORT_RUN = "194b8ba55f8844e1b7057b44869f4685"


def config(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text("utf-8")))


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text("utf-8")))


@contextmanager
def atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), "utf-8")


def parquet(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as temporary:
        frame.to_parquet(temporary, index=False)


def require_baseline(root: Path) -> dict[str, Any]:
    path = root / "baseline" / "verified.json"
    predictions = path.parent / "predictions.parquet"
    if not path.is_file() or not predictions.is_file():
        raise RuntimeError("Run the Week 4 baseline command successfully first.")
    record = read_json(path)
    if (
        record["split_sha256"]
        != "cc60d7a8a5524628241cff0e898238a2d9867b63641435dd6934969a5a34c11e"
        or record["months"] != 95
        or f"{record['mean_rank_ic']:.5f}" != "0.00394"
        or file_sha256(predictions) != record["predictions_sha256"]
    ):
        raise ValueError("Week 4 baseline gate failed. Re-run the baseline command.")
    return record


@contextmanager
def stage_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=0)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        yield


@contextmanager
def trial(name: str, params: dict[str, Any]):
    """Serialize numbering across Week 4 entry points, including failed trials."""
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Existing MLflow database is missing: {DATABASE}")
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    with stage_lock(DATABASE.parent / "week4_trial_lock.db"):
        mlflow = import_module("mlflow")
        mlflow.set_tracking_uri("sqlite:///" + DATABASE.resolve().as_posix())
        mlflow.set_experiment("ingen_qr_week4")
        with closing(sqlite3.connect(DATABASE)) as db:
            previous = db.execute(
                "SELECT MAX(CAST(p.value AS INTEGER)) FROM params p "
                "JOIN runs r ON r.run_uuid = p.run_uuid "
                "JOIN experiments e ON e.experiment_id = r.experiment_id "
                "WHERE p.key = 'cumulative_trial' AND "
                "e.name IN ('ingen_qr_week3', 'ingen_qr_week4')"
            ).fetchone()[0]
        number = max(1442, previous or 0) + 1
        with mlflow.start_run(
            run_name=name,
            tags={
                "run_kind": "trial",
                "research_only": "true",
                "holdout_locked": "true",
            },
        ):
            mlflow.log_param("cumulative_trial", number)
            mlflow.log_params(params)
            started = perf_counter()
            yield mlflow
            mlflow.log_metric("elapsed_seconds", perf_counter() - started)


def log_metrics(mlflow: Any, metrics: dict[str, Any]) -> None:
    mlflow.log_metrics(
        {
            key: float(value)
            for key, value in metrics.items()
            if value is not None and np.isfinite(value)
        }
    )


def export_trials(path: Path) -> None:
    """Export the frozen Week 3 history followed by Week 4 attempts."""
    mlflow = import_module("mlflow")
    mlflow.set_tracking_uri("sqlite:///" + DATABASE.resolve().as_posix())
    current = mlflow.search_runs(
        experiment_names=["ingen_qr_week4"],
        filter_string="tags.run_kind = 'trial'",
    )
    if not current.empty:
        current = (
            current.assign(_trial=current["params.cumulative_trial"].astype(int))
            .sort_values("_trial")
            .drop(columns="_trial")
        )
    prior = pd.read_csv("docs/W03_Experiment_Log.csv")
    with atomic_path(path) as temporary:
        pd.concat([prior, current], ignore_index=True).to_csv(temporary, index=False)
