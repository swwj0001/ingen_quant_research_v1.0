"""One OOS prediction per stock/month, shared samples, no model tuning."""

from pathlib import Path
from typing import Any

import pandas as pd

from ingen_qr.evaluation.signal_metrics import monthly_metrics, summarize
from ingen_qr.evaluation.splitters import PurgedWalkForward, training_indices
from ingen_qr.models.linear import OLS_FEATURES, ols_predict


def run_benchmarks(
    panel: pd.DataFrame,
    processed: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = panel[["date", "ticker", "price_at_forecast", *OLS_FEATURES]].merge(
        labels, on=["date", "ticker"], validate="one_to_one"
    )
    joined = joined.loc[
        (joined["label_status"] == "ok") & joined["price_at_forecast"]
    ].copy()
    # A wholly unavailable OLS characteristic is not silently replaced by an intercept.
    for name in OLS_FEATURES:
        counts = joined.groupby("date")[name].transform("count")
        joined = joined.loc[counts > 0]
    joined = joined.reset_index(drop=True)
    matrix = joined[["date", "ticker"]].merge(
        processed, on=["date", "ticker"], validate="one_to_one"
    )
    if len(joined) != len(matrix):
        raise ValueError("Preprocessing changed benchmark sample membership.")
    splitter = PurgedWalkForward(
        first_test=str(config["first_test_month"]),
        last_test=str(config["last_test_month"]),
    )
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    minimum = int(config["minimum_train_rows"])
    for fold in splitter.split(joined):
        train, test = joined.iloc[fold.refit], joined.iloc[fold.test]
        folds.append(
            {
                "test_month": fold.test_month,
                "training_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "refit_rows": len(train),
                "test_rows": len(test),
                "latest_training_label_end": (
                    train["label_end"].max() if len(train) else pd.NaT
                ),
            }
        )
        if len(train) < minimum:
            continue
        scored = test[["date", "ticker", "label_excess_return", "label_end"]].copy()
        scored["prediction"] = ols_predict(
            matrix.iloc[fold.refit],
            train["label_excess_return"],
            matrix.iloc[fold.test],
        )
        scored["model"] = "ols3"
        predictions.append(scored)
        momentum = scored.copy()
        momentum["prediction"] = matrix.iloc[fold.test]["momentum_252_21d"].to_numpy()
        momentum["model"] = "momentum"
        predictions.append(momentum)
    if not predictions:
        raise ValueError(
            "No valid OOS benchmark folds. Inspect CIK, price, fundamental coverage and label diagnostics; no placeholder results were created."
        )
    output = pd.concat(predictions, ignore_index=True)
    if output.duplicated(["model", "date", "ticker"]).any():
        raise ValueError("A stock/month received multiple OOS predictions.")
    pd.DataFrame(folds).to_csv(directory / "walk_forward_folds.csv", index=False)
    output.to_csv(directory / "benchmark_predictions.csv", index=False)
    metrics = monthly_metrics(output)
    if metrics.empty:
        raise ValueError(
            "No monthly metrics: at least 10 stocks and non-constant signals/labels are required."
        )
    metrics.to_csv(directory / "benchmark_monthly_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(directory / "benchmark_summary.csv", index=False)
    # Untuned 2015 calibration is recorded separately from 2016-2023 OOS results.
    calibration = []
    for month, test in joined.loc[
        joined["date"].between("2015-01-01", "2015-12-31")
    ].groupby("date"):
        idx = training_indices(
            joined, pd.Timestamp(str(month)).replace(day=1), test["forecast_at"].min()
        )
        if len(idx) < minimum:
            continue
        prediction = ols_predict(
            matrix.iloc[idx],
            joined.iloc[idx]["label_excess_return"],
            matrix.loc[test.index],
        )
        rows = test[["date", "ticker", "label_excess_return"]].copy()
        rows["model"], rows["prediction"] = "ols3", prediction
        calibration.append(rows)
    if calibration:
        monthly_metrics(pd.concat(calibration)).to_csv(
            directory / "calibration_2015_metrics.csv", index=False
        )
    return output, metrics, summary
