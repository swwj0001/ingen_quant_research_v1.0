"""Out-of-sample Week 3 statistics on aligned stock/date predictions."""

from importlib import import_module
from math import erfc, sqrt

import numpy as np
import pandas as pd

from ingen_qr.evaluation.signal_metrics import monthly_metrics, newey_west
from ingen_qr.evaluation.week3_samples import KEYS


def deciles(score: pd.Series) -> pd.Series:
    ranks = (score.rank(method="average") - 0.5) / score.notna().sum()
    return pd.Series(np.ceil(ranks.to_numpy() * 10.0), index=ranks.index)


def extended_monthly(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    buckets = []
    for model, group in scores.groupby("model", sort=True):
        previous = None
        for date, month in group.groupby("date", sort=True):
            month = month.dropna(subset=["prediction", "label_excess_return"])
            bucket = deciles(month["prediction"])
            by_bucket = month["label_excess_return"].groupby(bucket).mean()
            for number, value in by_bucket.items():
                buckets.append(
                    {
                        "model": model,
                        "date": date,
                        "decile": number,
                        "mean_return": value,
                    }
                )
            mono = (
                by_bucket.rank().corr(
                    pd.Series(np.arange(len(by_bucket)), index=by_bucket.index)
                )
                if len(by_bucket) >= 3 and by_bucket.nunique() > 1
                else np.nan
            )

            top = month.loc[bucket.eq(10), "ticker"]
            bottom = month.loc[bucket.eq(1), "ticker"]
            turnover = np.nan
            if len(top) and len(bottom):
                weights = pd.concat(
                    [
                        pd.Series(0.5 / len(top), index=top),
                        pd.Series(-0.5 / len(bottom), index=bottom),
                    ]
                )
                if previous is not None:
                    turnover = (
                        float(weights.sub(previous, fill_value=0).abs().sum()) / 2.0
                    )
                previous = weights

            score = month["prediction"]
            target = month["label_excess_return"]
            score_residual = score - score.groupby(month["industry"]).transform("mean")
            target_residual = target - target.groupby(month["industry"]).transform(
                "mean"
            )
            industry_ic = (
                score_residual.corr(target_residual)
                if len(month) >= 10
                and score_residual.nunique() > 1
                and target_residual.nunique() > 1
                else np.nan
            )
            records.append(
                {
                    "model": model,
                    "date": date,
                    "decile_monotonicity": mono,
                    "turnover": turnover,
                    "industry_neutral_ic": industry_ic,
                    "sic_coverage": month["sic_available"].mean(),
                }
            )

    extra = pd.DataFrame(records)
    main = monthly_metrics(scores).merge(extra, on=["model", "date"], how="left")
    return main, pd.DataFrame(buckets)


def classification(scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, frame in scores.groupby("model"):
        ranks = frame.groupby("date")["prediction"].rank(method="average")
        counts = frame.groupby("date")["prediction"].transform("count")
        percentile = (ranks - 0.5) / counts
        calls = pd.Series(np.nan, index=frame.index)
        calls.loc[percentile <= 1 / 3] = 0.0
        calls.loc[percentile >= 2 / 3] = 1.0
        labeled = frame["label_class"].notna() & frame["prediction"].notna()
        covered = labeled & calls.notna()
        truth, prediction = frame.loc[covered, "label_class"], calls.loc[covered]
        tp = int((truth.eq(1) & prediction.eq(1)).sum())
        fp = int((truth.eq(0) & prediction.eq(1)).sum())
        fn = int((truth.eq(1) & prediction.eq(0)).sum())
        actual = frame.loc[labeled, "label_class"]
        auc = np.nan
        if actual.nunique() == 2:
            auc = import_module("sklearn.metrics").roc_auc_score(
                actual, frame.loc[labeled, "prediction"]
            )
        records.append(
            {
                "model": model,
                "classification_n": int(labeled.sum()),
                "classification_coverage": (
                    float(covered.sum() / labeled.sum()) if labeled.any() else np.nan
                ),
                "accuracy": (truth == prediction).mean() if len(truth) else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                "auc": auc,
            }
        )
    return pd.DataFrame(records)


def decay(scores: pd.DataFrame, horizons: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for length, labels in horizons.items():
        merged = scores.drop(columns=["label_excess_return", "label_end"]).merge(
            labels, on=KEYS, validate="many_to_one"
        )
        if merged.empty:
            continue
        frames.append(
            monthly_metrics(merged)[["date", "model", "n", "ic", "rank_ic"]].assign(
                horizon_sessions=length, forward_month=length // 21
            )
        )
    return pd.concat(frames, ignore_index=True)


def summary(metrics: pd.DataFrame, lags: int) -> pd.DataFrame:
    records = []
    for model, frame in metrics.groupby("model"):
        frame = frame.sort_values("date")
        values = frame["rank_ic"]
        deviation = values.std(ddof=1)
        average = frame[
            [
                "ic",
                "rank_ic",
                "decile_spread",
                "decile_monotonicity",
                "turnover",
                "industry_neutral_ic",
                "sic_coverage",
            ]
        ].mean()
        records.append(
            {
                "model": model,
                "months": len(frame),
                **average.to_dict(),
                "icir": values.mean() / deviation if deviation > 0 else np.nan,
                "nw_t_ic": newey_west(frame["ic"], lags),
                "nw_t_rank_ic": newey_west(values, lags),
            }
        )
    return pd.DataFrame(records)


def paired_newey_west(metrics: pd.DataFrame, lags: int) -> pd.DataFrame:
    reference = metrics.loc[metrics["model"] == "ols3", ["date", "ic", "rank_ic"]]
    records = []
    for model, frame in metrics.loc[metrics["model"] != "ols3"].groupby("model"):
        pairs = frame.merge(reference, on="date", suffixes=("", "_ols3")).sort_values(
            "date"
        )
        for metric in ("ic", "rank_ic"):
            delta = (pairs[metric] - pairs[metric + "_ols3"]).dropna()
            t = newey_west(delta, lags)
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "paired_months": len(delta),
                    "mean_difference": delta.mean(),
                    "nw_t": t,
                    "p_two_sided_asymptotic": (
                        erfc(abs(t) / sqrt(2)) if np.isfinite(t) else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)
