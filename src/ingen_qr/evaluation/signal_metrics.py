# mypy: disable-error-code=import-untyped

"""Monthly signal statistics; fixed six-lag Newey-West inference."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west(values: pd.Series, lags: int = 6) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) < max(lags + 2, 12) or np.std(clean) == 0:
        return float("nan")
    fit = sm.OLS(clean, np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(fit.tvalues[0])


def monthly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, date), frame in predictions.groupby(["model", "date"]):
        group = frame.dropna(subset=["prediction", "label_excess_return"])
        if (
            len(group) < 10
            or group["prediction"].nunique() < 2
            or group["label_excess_return"].nunique() < 2
        ):
            continue
        prediction, target = group["prediction"], group["label_excess_return"]
        ranks = (prediction.rank(method="average") - 0.5) / len(group)
        top, bottom = target.loc[ranks >= 0.9], target.loc[ranks <= 0.1]
        target_ranks = (target.rank(method="average") - 0.5) / len(group)
        records.append(
            {
                "model": model,
                "date": date,
                "n": len(group),
                "ic": prediction.corr(target),
                "rank_ic": prediction.rank().corr(target.rank()),
                "hit_rate": (
                    float((np.sign(prediction) == np.sign(target)).mean())
                    if model == "ols3"
                    else np.nan
                ),
                "rank_hit_rate": float(((ranks > 0.5) == (target_ranks > 0.5)).mean()),
                "decile_spread": (
                    top.mean() - bottom.mean() if len(top) and len(bottom) else np.nan
                ),
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "model",
            "date",
            "n",
            "ic",
            "rank_ic",
            "hit_rate",
            "rank_hit_rate",
            "decile_spread",
        ],
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in metrics.groupby("model"):
        rank_ic = group["rank_ic"]
        sd = rank_ic.std(ddof=1)
        records.append(
            {
                "model": model,
                "months": len(group),
                "mean_ic": group["ic"].mean(),
                "mean_rank_ic": rank_ic.mean(),
                "icir": rank_ic.mean() / sd if sd > 0 else np.nan,
                "nw_t_rank_ic": newey_west(rank_ic),
                "nw_t_ic": newey_west(group["ic"]),
                "mean_hit_rate": group["hit_rate"].mean(),
                "mean_rank_hit_rate": group["rank_hit_rate"].mean(),
                "mean_decile_spread": group["decile_spread"].mean(),
                "nw_t_decile_spread": newey_west(group["decile_spread"]),
            }
        )
    return pd.DataFrame(records)
