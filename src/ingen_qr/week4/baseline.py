"""Verify and cache the Week 3 random forest without any tuning."""

import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.evaluation.signal_metrics import monthly_metrics
from ingen_qr.evaluation.week3_samples import (
    Week3Sample,
    file_sha256,
    load_verified_sample,
    read_baseline,
)
from ingen_qr.models.week3 import Regressor
from ingen_qr.week4.common import (
    DATABASE,
    KEYS,
    WEEK2,
    WEEK3,
    WEEK3_REPORT_RUN,
    atomic_path,
    config,
    parquet,
    require_baseline,
    stage_lock,
    trial,
    write_json,
)

SPLIT = "cc60d7a8a5524628241cff0e898238a2d9867b63641435dd6934969a5a34c11e"
MONTHS = tuple(pd.period_range("2016-01", "2023-11", freq="M").astype(str))
LOG = Path("docs/W03_Experiment_Log.csv")
SPEC = {
    "n_estimators": int,
    "max_depth": int,
    "min_samples_leaf": int,
    "max_features": float,
    "n_jobs": int,
}
PREDICTION_COLUMNS = [*KEYS, "label_excess_return", "label_end", "label_class"]


def annual_parameters(log: Path = LOG) -> dict[int, dict[str, Any]]:
    frame = pd.read_csv(log)
    frame = frame.loc[
        frame["params.model"].eq("random_forest")
        & frame["tags.mlflow.parentRunId"].eq(WEEK3_REPORT_RUN)
        & frame["tags.run_kind"].eq("trial")
    ]
    months = {f"{year}-01" for year in range(2016, 2024)}
    if (
        set(frame["params.tuning_month"]) != months
        or set(frame["params.split_sha256"]) != {SPLIT}
        or set(frame["params.seed"]) != {1}
        or set(frame["params.model.random_state"]) != {1}
        or set(frame["params.week3_config_sha256"]) != {file_sha256(WEEK3)}
    ):
        raise ValueError("Week 3 experiment log does not match the frozen RF run.")
    result = {}
    for month, group in frame.groupby("params.tuning_month"):
        if sorted(group["params.trial_number"].tolist()) != list(range(30)):
            raise ValueError(f"Incomplete Week 3 random forest search for {month}.")
        winner = (
            group.loc[
                group["status"].eq("FINISHED")
                & np.isfinite(group["metrics.validation_rank_ic"])
            ]
            .sort_values(
                ["metrics.validation_rank_ic", "params.trial_number"],
                ascending=[False, True],
            )
            .iloc[0]
        )
        result[int(str(month)[:4])] = {
            name: cast(winner["params.model." + name]) for name, cast in SPEC.items()
        }
    return result


def sample() -> Week3Sample:
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Existing Week 3 input is missing: {DATABASE}")
    frozen = read_baseline(DATABASE)
    data = load_verified_sample(WEEK2, frozen)
    if frozen.split_sha256 != SPLIT or data.split_sha256 != SPLIT:
        raise ValueError("Week 3 split SHA-256 mismatch.")
    if tuple(f"{fold.test_month:%Y-%m}" for fold in data.folds) != MONTHS:
        raise ValueError("Week 3 does not have the expected 95 monthly folds.")
    if len(data.matrix.columns) != config(WEEK3)["expected_model_columns"]:
        raise ValueError("Week 3 model columns changed.")
    return data


def assess(frame: pd.DataFrame, expected: pd.DataFrame | None = None) -> float:
    scores = frame.loc[frame["model"].eq("random_forest")].sort_values(KEYS)
    if scores.duplicated(KEYS).any() or not np.isfinite(scores["prediction"]).all():
        raise ValueError("Week 3 RF predictions contain duplicates or invalid values.")
    if expected is not None:
        columns = [*KEYS, "label_excess_return", "label_end"]
        pd.testing.assert_frame_equal(
            scores[columns].reset_index(drop=True),
            expected[columns].sort_values(KEYS).reset_index(drop=True),
            check_dtype=False,
        )
    monthly = monthly_metrics(scores)
    if monthly.empty or tuple(monthly["date"].dt.to_period("M").astype(str)) != MONTHS:
        raise ValueError(
            f"Week 3 RF Rank IC must exist for exactly 95 months, got {len(monthly)}."
        )
    value = float(monthly["rank_ic"].mean())
    if f"{value:.5f}" != "0.00394":
        raise ValueError(
            f"Week 3 RF mean Rank IC mismatch: {value:.10f}, expected 0.00394."
        )
    return value


def verify(root: Path, saved_path: Path | None = None) -> None:
    directory = root / "baseline"
    saved_path = saved_path or (
        Path(config(WEEK3)["output_directory"])
        / WEEK3_REPORT_RUN
        / "predictions.parquet"
    )
    with stage_lock(directory / "baseline_lock.db"):
        local = pd.read_parquet(saved_path) if saved_path.is_file() else None
        if local is not None:
            print(f"Saved Week 3 mean Rank IC: {assess(local):.10f}", flush=True)
        elif (
            saved_path
            != Path(config(WEEK3)["output_directory"])
            / WEEK3_REPORT_RUN
            / "predictions.parquet"
        ):
            raise FileNotFoundError(f"Saved Week 3 predictions not found: {saved_path}")
        else:
            print(
                "Saved Week 3 predictions absent. Verifying with a refit.", flush=True
            )
        data = sample()
        expected = pd.concat([data.meta.iloc[fold.test] for fold in data.folds])
        if local is not None:
            assess(local, expected)
        annual = annual_parameters()
        signature = hashlib.sha256(
            json.dumps([data.hashes, annual, SPLIT], sort_keys=True).encode()
        ).hexdigest()
        marker = directory / "verified.json"
        if marker.is_file():
            record = require_baseline(root)
            if record["signature"] == signature and all(
                (directory / signature / f"{month}.joblib").is_file()
                for month in MONTHS
            ):
                cached = pd.read_parquet(directory / "predictions.parquet")
                print("Expected baseline fits: 0 (verified cache).", flush=True)
                print(
                    f"PASS {SPLIT}, 95 months, mean Rank IC {assess(cached, expected):.10f}"
                )
                return
        marker.unlink(missing_ok=True)
        cache = directory / signature
        joblib = import_module("joblib")
        remaining = sum(
            not (
                (cache / f"{month}.parquet").is_file()
                and (cache / f"{month}.joblib").is_file()
            )
            for month in MONTHS
        )
        print(f"Expected baseline fits: {remaining} (95 total, no tuning).", flush=True)
        with trial(
            "week4_baseline_rf", {"split_sha256": SPLIT, "planned_fits": remaining}
        ) as mlflow:
            frames = []
            for month, fold in zip(MONTHS, data.folds, strict=True):
                prediction_path = cache / f"{month}.parquet"
                model_path = cache / f"{month}.joblib"
                train = data.meta.iloc[fold.refit]
                test = data.meta.iloc[fold.test]
                if prediction_path.is_file() and model_path.is_file():
                    frames.append(pd.read_parquet(prediction_path))
                    continue
                print(f"Baseline RF {month}", flush=True)
                model = Regressor("random_forest", annual[fold.test_month.year], 1)
                model.fit(
                    data.matrix.iloc[fold.refit], train["label_excess_return"], train
                )
                result = test[PREDICTION_COLUMNS].assign(
                    model="random_forest",
                    prediction=model.predict(data.matrix.iloc[fold.test]),
                )
                with atomic_path(model_path) as temporary:
                    joblib.dump(model, temporary, compress=3)
                parquet(result, prediction_path)
                frames.append(result)
            result = pd.concat(frames, ignore_index=True)
            value = assess(result, expected)
            destination = directory / "predictions.parquet"
            parquet(result, destination)
            monthly_metrics(result).to_csv(
                directory / "monthly_metrics.csv", index=False
            )
            pd.DataFrame.from_dict(annual, orient="index").rename_axis("year").to_csv(
                directory / "annual_parameters.csv"
            )
            mlflow.log_metrics({"mean_rank_ic": value, "months": len(MONTHS)})
            write_json(
                {
                    "split_sha256": SPLIT,
                    "months": len(MONTHS),
                    "mean_rank_ic": value,
                    "signature": signature,
                    "predictions_sha256": file_sha256(destination),
                },
                marker,
            )
        print(f"PASS {SPLIT}, 95 months, mean Rank IC {value:.10f}")
