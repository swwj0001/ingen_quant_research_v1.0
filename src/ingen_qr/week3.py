"""Train the Week 3 model families against the frozen Week 2 sample."""

import argparse
import os
import sqlite3
from contextlib import closing
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from ingen_qr.evaluation.splitters import Fold
from ingen_qr.evaluation.week3_samples import (
    BASELINE_RUN_ID,
    Week3Sample,
    file_sha256,
    load_verified_sample,
    read_baseline,
)
from ingen_qr.models.week3 import Model, Regressor

SMOKE_EXPERIMENT = "ingen_qr_week3_smoke"


def validation_rank_ic(meta: pd.DataFrame, scores: NDArray[np.float64]) -> float:
    frame = meta[["date", "label_excess_return"]].assign(score=scores)
    values = []
    for _, month in frame.groupby("date"):
        if len(month) >= 10:
            values.append(
                month["score"].rank().corr(month["label_excess_return"].rank())
            )
    valid = np.asarray(values, dtype=float)
    return (
        float(valid[np.isfinite(valid)].mean()) if np.isfinite(valid).any() else np.nan
    )


def score_validation(model: Model, sample: Week3Sample, fold: Fold) -> float:
    train = sample.meta.iloc[fold.train]
    model.fit(sample.matrix.iloc[fold.train], train["label_excess_return"], train)
    return validation_rank_ic(
        sample.meta.iloc[fold.validation],
        model.predict(sample.matrix.iloc[fold.validation]),
    )


def select_trial_params(trial: Any, spec: dict[str, Any]) -> dict[str, Any]:
    params = dict(spec["fixed"])
    for name, bounds in spec["search"].items():
        kind = bounds["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, bounds["choices"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, bounds["low"], bounds["high"])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, bounds["low"], bounds["high"], log=bounds.get("log", False)
            )
        else:
            raise ValueError(f"Unsupported parameter type: {kind}")
    return params


class Trials:
    def __init__(
        self,
        mlflow: Any,
        experiment_id: str,
        hashes: dict[str, str],
        baseline_trials: int = 2,
    ) -> None:
        self.mlflow = mlflow
        self.hashes = hashes
        previous = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string="tags.run_kind = 'trial'"
        )
        self.count = baseline_trials + len(previous)

    def tune(
        self,
        name: str,
        spec: dict[str, Any],
        seed: int,
        budget: int,
        sample: Week3Sample,
        fold: Fold,
    ) -> dict[str, Any]:
        optuna = import_module("optuna")
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )

        def objective(trial: Any) -> float:
            self.count += 1
            with self.mlflow.start_run(
                run_name=f"{name}-{fold.test_month:%Y-%m}-{trial.number}",
                nested=True,
                tags={"run_kind": "trial"},
            ):
                self.mlflow.log_params(
                    {
                        **self.hashes,
                        "model": name,
                        "tuning_month": f"{fold.test_month:%Y-%m}",
                        "trial_number": trial.number,
                        "cumulative_trial": self.count,
                        "seed": seed,
                    }
                )
                params = select_trial_params(trial, spec)
                model = Regressor(name, params, seed)
                self.mlflow.log_params(
                    {f"model.{key}": value for key, value in model.get_params().items()}
                )
                score = score_validation(model, sample, fold)
                if not np.isfinite(score):
                    raise optuna.TrialPruned("No defined monthly validation Rank IC.")
                self.mlflow.log_metric("validation_rank_ic", score)
                return score

        study.optimize(objective, n_trials=budget)
        return {**spec["fixed"], **study.best_params}


def predict_fold(
    sample: Week3Sample,
    fold: Fold,
    validation: dict[str, float],
    params: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    train = sample.meta.iloc[fold.refit]
    test = sample.meta.iloc[fold.test]
    output = test[["date", "ticker", "label_excess_return", "label_end", "label_class"]]
    scores = {}
    for name, options in params.items():
        model = Regressor(name, options, seed)
        model.fit(sample.matrix.iloc[fold.refit], train["label_excess_return"], train)
        scores[name] = model.predict(sample.matrix.iloc[fold.test])
        if not np.isfinite(scores[name]).all():
            raise ValueError(f"{name} produced missing or non-finite OOS scores.")

    names = sorted(
        [name for name, score in validation.items() if np.isfinite(score)],
        key=lambda name: (-validation[name], name),
    )[:3]
    if len(names) != 3:
        raise ValueError("Ensembling needs three defined fold validation Rank ICs.")
    ranks = pd.DataFrame({name: scores[name] for name in names}, index=test.index)
    ranks = ranks.groupby(test["date"]).rank(method="average", pct=True)
    scores["ensemble_top3"] = ranks.mean(axis=1).to_numpy(dtype=float)
    result = pd.concat(
        [output.assign(model=name, prediction=value) for name, value in scores.items()],
        ignore_index=True,
    )
    return result, names


def _run(config_path: Path, smoke: bool) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    budget = 2 if smoke else int(config["trials_per_model"])
    experiment_name = SMOKE_EXPERIMENT if smoke else str(config["experiment"])
    output_directory = Path("data/week3_smoke" if smoke else config["output_directory"])
    if not smoke and experiment_name == SMOKE_EXPERIMENT:
        raise ValueError("The smoke experiment name is reserved for --smoke.")
    database = Path("mlruns/tracking.db")
    baseline = read_baseline(database)
    sample = load_verified_sample(Path("configs/week2.yaml"), baseline)
    if baseline.split_sha256 != sample.split_sha256:
        raise ValueError("Run the baseline split backfill before Week 3 training.")
    if len(sample.matrix.columns) != int(config["expected_model_columns"]):
        raise ValueError(f"Unexpected model column count: {len(sample.matrix.columns)}")
    if budget < 1:
        raise ValueError("trials_per_model must be positive.")
    folds = sample.folds[:2] if smoke else sample.folds

    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    mlflow = import_module("mlflow")
    mlflow.set_tracking_uri("sqlite:///" + database.resolve().as_posix())
    experiment = mlflow.set_experiment(experiment_name)
    hashes = {
        **sample.hashes,
        "split_sha256": sample.split_sha256,
        "week3_config_sha256": file_sha256(config_path),
    }
    trials = Trials(
        mlflow, experiment.experiment_id, hashes, baseline_trials=0 if smoke else 2
    )

    with mlflow.start_run(
        run_name="week3_models",
        tags={"run_kind": "models", "run_mode": "smoke" if smoke else "formal"},
    ) as active:
        if active.info.start_time <= baseline.end_time:
            raise ValueError("Week 3 run must start after the Week 2 baseline.")
        mlflow.log_params(
            {
                **hashes,
                "baseline_run_id": BASELINE_RUN_ID,
                "seed": seed,
                "trials_per_model": budget,
                "feature_columns": len(sample.matrix.columns),
                "evaluated_folds": len(folds),
                "cumulative_trial_start": trials.count,
            }
        )
        predictions = []
        selections: list[dict[str, object]] = []
        params: dict[str, dict[str, Any]] = {}
        for fold in folds:
            month = fold.test_month
            print(f"Week 3 {month:%Y-%m}", flush=True)
            if not params or (not smoke and month.month == 1):
                params = {
                    name: trials.tune(name, spec, seed, budget, sample, fold)
                    for name, spec in config["models"].items()
                }
            validation = {
                name: score_validation(Regressor(name, options, seed), sample, fold)
                for name, options in params.items()
            }
            scored, names = predict_fold(sample, fold, validation, params, seed)
            predictions.append(scored)
            selections.extend(
                {
                    "test_month": month,
                    "model": name,
                    "validation_rank_ic": value,
                    "selected_for_ensemble": name in names,
                }
                for name, value in validation.items()
            )

        output = output_directory / active.info.run_id
        output.mkdir(parents=True)
        pd.concat(predictions, ignore_index=True).to_parquet(
            output / "predictions.parquet", index=False
        )
        pd.DataFrame(selections).to_csv(output / "validation_by_fold.csv", index=False)
        sample.manifest.to_csv(output / "walk_forward_folds.csv", index=False)
        mlflow.log_param("cumulative_trial_end", trials.count)
        mlflow.log_artifacts(str(output), artifact_path="week3")
        return output


def run(config_path: Path, *, smoke: bool = False) -> Path:
    # Serialize trainers so simultaneous invocations cannot reuse trial numbers.
    if not Path("mlruns/tracking.db").is_file():
        raise FileNotFoundError("Expected the existing mlruns/tracking.db database.")
    lock_path = (
        "mlruns/week3_smoke_training_lock.db"
        if smoke
        else "mlruns/week3_training_lock.db"
    )
    with closing(sqlite3.connect(lock_path, timeout=0)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        return _run(config_path, smoke)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/week3.yaml"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first two eligible folds with two trials per model family.",
    )
    args = parser.parse_args()
    print("Week 3 output:", run(args.config, smoke=args.smoke))
