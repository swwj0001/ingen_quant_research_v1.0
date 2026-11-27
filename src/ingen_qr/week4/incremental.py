"""Compare two RF augmentations and raw single signals on the frozen sample."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.evaluation.signal_metrics import monthly_metrics
from ingen_qr.evaluation.week3_samples import Week3Sample, file_sha256
from ingen_qr.features.preprocessing import preprocess
from ingen_qr.models.linear import OLS_FEATURES
from ingen_qr.models.week3 import Regressor
from ingen_qr.week4 import baseline
from ingen_qr.week4.common import (
    KEYS,
    WEEK2,
    config,
    export_trials,
    log_metrics,
    parquet,
    read_json,
    trial,
    write_json,
)
from ingen_qr.week4.paired import paired_monthly
from ingen_qr.week4.reports import text_report
from ingen_qr.week4.text_features import FEATURES, SENTIMENT_CHANGES

SHORT_MOMENTUM = "reversal_21d"


def model_matrix(
    data: Week3Sample, features: pd.DataFrame, names: list[str]
) -> pd.DataFrame:
    if not features.index.equals(data.meta.index) or not features.index.equals(
        data.matrix.index
    ):
        raise ValueError("Text features changed the Week 3 sample alignment.")
    raw = features[names].replace([np.inf, -np.inf], np.nan)
    flags = raw.isna().astype(float).add_prefix("missing_")
    panel = pd.concat([data.meta[KEYS], raw], axis=1)
    normalized, _ = preprocess(panel, names)
    return pd.concat([data.matrix, normalized[names], flags], axis=1)


def fit_variant(
    data: Week3Sample, features: pd.DataFrame, names: list[str], target: Path
) -> pd.DataFrame:
    matrix = model_matrix(data, features, names)
    frames = []
    annual = baseline.annual_parameters()
    for fold in data.folds:
        month = f"{fold.test_month:%Y-%m}"
        path = target / f"{month}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path))
            continue
        print(f"{target.parent.name} RF {month}", flush=True)
        model = Regressor("random_forest", annual[fold.test_month.year], 1)
        train = data.meta.iloc[fold.refit]
        test = data.meta.iloc[fold.test]
        model.fit(matrix.iloc[fold.refit], train["label_excess_return"], train)
        frame = test[baseline.PREDICTION_COLUMNS].assign(
            model=target.parent.name,
            prediction=model.predict(matrix.iloc[fold.test]),
        )
        parquet(frame, path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def correlate(
    frame: pd.DataFrame, feature: str, response: str, controls: list[str]
) -> float:
    panel = (
        frame[[feature, response, *controls]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if (
        len(panel) < max(10, len(controls) + 3)
        or panel[feature].nunique() < 2
        or panel[response].nunique() < 2
    ):
        return np.nan
    ranks = panel.rank()
    outcomes = ranks[[feature, response]].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(panel)), ranks[controls].to_numpy(dtype=float)])
    residuals = outcomes - x @ np.linalg.lstsq(x, outcomes, rcond=None)[0]
    if min(residuals.std(axis=0)) < 1e-10:
        return np.nan
    return float(np.corrcoef(residuals.T)[0, 1])


def partial_correlations(oos: pd.DataFrame, outputs: Path) -> None:
    raw_path = Path(config(WEEK2)["output_directory"]) / "features_raw.parquet"
    price = pd.read_parquet(raw_path, columns=[*KEYS, SHORT_MOMENTUM, *OLS_FEATURES])
    frame = oos.merge(price, on=KEYS, validate="one_to_one")
    results = [
        {
            "date": date,
            "feature": name,
            "text_observed": int(group[name].notna().sum()),
            "text_return_partial_rank": correlate(
                group,
                name,
                "label_excess_return",
                [SHORT_MOMENTUM, *OLS_FEATURES],
            ),
            "text_short_momentum_partial_rank": correlate(
                group, name, SHORT_MOMENTUM, OLS_FEATURES
            ),
        }
        for date, group in frame.groupby("date")
        for name in FEATURES
    ]
    pd.DataFrame(results).to_csv(outputs / "partial_correlations.csv", index=False)


def rank_series(metrics: pd.DataFrame) -> pd.Series:
    return metrics.set_index("date")["rank_ic"]


def summary(
    name: str, metrics: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, Any]:
    values = rank_series(metrics)
    return {
        "variant": name,
        "months": len(values),
        "mean_rank_ic": float(values.mean()),
        **paired_monthly(values, rank_series(reference)),
    }


def run(settings: dict[str, Any], root: Path, gate: dict[str, Any]) -> None:
    corpus = root / "full"
    path = corpus / "firm_month_features.parquet"
    receipt_path = corpus / "completed.json"
    if not receipt_path.is_file():
        raise RuntimeError("Run the full EDGAR corpus stage first.")
    receipt = read_json(receipt_path)
    if (
        receipt["mode"] != "full"
        or receipt["baseline_signature"] != gate["signature"]
        or receipt["features_sha256"] != file_sha256(path)
        or any(
            receipt["text_config"].get(key) != value for key, value in settings.items()
        )
    ):
        raise ValueError(
            "The full corpus receipt does not match the baseline or text settings."
        )
    raw = pd.read_parquet(path)
    data = baseline.sample()
    pd.testing.assert_frame_equal(
        raw[KEYS].sort_values(KEYS).reset_index(drop=True),
        data.meta[KEYS].sort_values(KEYS).reset_index(drop=True),
        check_dtype=False,
    )
    features = data.meta[KEYS].merge(raw, on=KEYS, validate="one_to_one")[FEATURES]
    features = features.replace([np.inf, -np.inf], np.nan)
    if not features[SENTIMENT_CHANGES].notna().any().any():
        raise ValueError(
            "No sentiment-change scores were observed. See firm_month_coverage.csv."
        )
    scores = pd.read_parquet(root / "baseline" / "predictions.parquet")
    reference = monthly_metrics(scores)
    base_scores = scores[[*KEYS, "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}
    )
    frame = raw.merge(
        scores[[*KEYS, "label_excess_return"]], on=KEYS, validate="one_to_one"
    )
    frame = frame.merge(base_scores, on=KEYS, validate="one_to_one")
    directory = root / "incremental"
    signature = hashlib.sha256(
        (
            receipt["features_sha256"]
            + gate["predictions_sha256"]
            + file_sha256(Path(__file__))
        ).encode()
    ).hexdigest()
    results = [summary("baseline", reference, reference.iloc[:0])]
    all_metrics = [reference]
    for name, names in (
        ("rf_all_text", FEATURES),
        ("rf_sentiment_change", SENTIMENT_CHANGES),
    ):
        path = directory / name / signature
        final = path / "predictions.parquet"
        missing = (
            0
            if final.is_file()
            else sum(
                not (path / f"{month}.parquet").is_file() for month in baseline.MONTHS
            )
        )
        print(f"Expected {name} fits: {missing} (95 total, no tuning).", flush=True)
        if final.is_file():
            metrics = monthly_metrics(pd.read_parquet(final))
            result = summary(name, metrics, reference)
        else:
            with trial(
                name,
                {
                    "split_sha256": baseline.SPLIT,
                    "planned_fits": missing,
                    "signal_columns": ",".join(names),
                    "features_sha256": receipt["features_sha256"],
                },
            ) as mlflow:
                predictions = fit_variant(data, features, names, path)
                metrics = monthly_metrics(predictions)
                result = summary(name, metrics, reference)
                if result["paired_months"] != len(baseline.MONTHS):
                    raise ValueError(f"{name} did not produce the full paired sample.")
                log_metrics(mlflow, {k: v for k, v in result.items() if k != "variant"})
                parquet(predictions, final)
        if result["paired_months"] != len(baseline.MONTHS):
            raise ValueError(f"{name} cache does not cover all 95 months.")
        results.append(result)
        all_metrics.append(metrics)
    print(
        f"Expected single-signal RF fits: 0 ({len(FEATURES)} raw rank IC variants).",
        flush=True,
    )
    for name in FEATURES:
        single = f"signal_{name}"
        observed = frame.loc[frame[name].notna()].assign(model=single)
        raw_predictions = observed.assign(prediction=observed[name])
        matching_baseline = observed.assign(
            model="matched_baseline", prediction=observed["baseline_prediction"]
        )
        computed = monthly_metrics(raw_predictions)
        result = summary(single, computed, monthly_metrics(matching_baseline))
        results.append(result)
        all_metrics.append(computed)
        completed = directory / "single_signals" / signature / f"{name}.json"
        if not completed.is_file():
            with trial(
                single,
                {
                    "split_sha256": baseline.SPLIT,
                    "planned_fits": 0,
                    "evaluation": "raw_signal",
                    "features_sha256": receipt["features_sha256"],
                },
            ) as mlflow:
                log_metrics(mlflow, {k: v for k, v in result.items() if k != "variant"})
            write_json({"logged": True}, completed)
    result_table = pd.DataFrame(results)
    path = directory / "incremental_summary.csv"
    result_table.to_csv(path, index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(
        directory / "monthly_rank_ic.csv", index=False
    )
    partial_correlations(frame, directory)
    h2 = result_table.loc[result_table["variant"].eq("rf_sentiment_change")].iloc[0]
    delta, nw = float(h2["mean_difference"]), float(h2["nw_t"])
    if not np.isfinite(nw):
        outcome = "H2 inconclusive because the Newey–West t-statistic is undefined."
    elif delta > 0 and nw > 2:
        outcome = f"H2 supported. Average paired Rank IC increment {delta:.6f}, Newey–West t {nw:.3f}."
    else:
        outcome = f"H2 not supported under the prespecified rule. Average paired Rank IC increment {delta:.6f}, Newey–West t {nw:.3f}."
    (directory / "H2.txt").write_text(outcome + "\n", "utf-8")
    text_report(directory, corpus, result_table, outcome)
    export_trials(root / "W04_Experiment_Log.csv")
    print(outcome, flush=True)
    print(f"Text results: {path}", flush=True)
