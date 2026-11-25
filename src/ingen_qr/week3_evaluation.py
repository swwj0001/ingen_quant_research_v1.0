"""Build Week 3's out-of-sample report, tables and charts."""

import argparse
import os
from importlib import import_module
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from ingen_qr.evaluation.reporting import markdown_table
from ingen_qr.evaluation.splitters import split_manifest_sha256
from ingen_qr.evaluation.week3_horizons import horizon_labels
from ingen_qr.evaluation.week3_industry import industry_snapshot
from ingen_qr.evaluation.week3_metrics import (
    classification,
    decay,
    extended_monthly,
    paired_newey_west,
    summary,
)
from ingen_qr.evaluation.week3_samples import (
    BASELINE_RUN_ID,
    KEYS,
    load_verified_sample,
    read_baseline,
)

INDUSTRY_NOTE = (
    "Industry-neutral IC uses the current SEC submissions SIC assigned to the "
    "Fama–French 12 groups. This is a current classification applied to historical "
    "observations, not a point-in-time industry classification. Missing SIC is "
    "assigned to Other. SIC coverage measures valid SEC SIC before the Other fallback."
)


def stability(
    metrics: pd.DataFrame, conditions: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    frame = metrics.merge(
        conditions[["date", "vix_level"]], on="date", validate="many_to_one", how="left"
    )
    year = frame["date"].dt.year
    frame["period"] = "pre-2020"
    frame.loc[year.between(2020, 2021), "period"] = "2020–2021"
    frame.loc[year.between(2022, 2023), "period"] = "2022–2023"
    frame["vix_regime"] = "missing"
    frame.loc[frame["vix_level"].le(threshold), "vix_regime"] = "low"
    frame.loc[frame["vix_level"].gt(threshold), "vix_regime"] = "high"
    return pd.concat(
        [
            frame.groupby(["model", axis], as_index=False)
            .agg(
                months=("ic", "size"),
                mean_ic=("ic", "mean"),
                mean_rank_ic=("rank_ic", "mean"),
            )
            .rename(columns={axis: "group"})
            .assign(axis=axis)
            for axis in ("period", "vix_regime")
        ],
        ignore_index=True,
    )


def export_trials(tracking_uri: str, experiment: str, table: Path) -> None:
    mlflow = import_module("mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="tags.run_kind = 'trial'",
        order_by=["attributes.start_time ASC", "attributes.run_id ASC"],
    )
    parent = mlflow.get_run(BASELINE_RUN_ID)
    baseline_rows = []
    for count, name in enumerate(("ols3", "momentum"), start=1):
        baseline_rows.append(
            {
                "run_id": BASELINE_RUN_ID,
                "tags.run_kind": "benchmark",
                "params.model": name,
                "params.cumulative_trial": count,
                "start_time": pd.to_datetime(
                    parent.info.start_time, unit="ms", utc=True
                ),
                "end_time": pd.to_datetime(parent.info.end_time, unit="ms", utc=True),
                "status": parent.info.status,
            }
        )
    pd.concat([pd.DataFrame(baseline_rows), runs], ignore_index=True).to_csv(
        table, index=False
    )


def chart_tables(
    monthly: pd.DataFrame, decay_table: pd.DataFrame, output: Path
) -> None:
    matplotlib = import_module("matplotlib")
    matplotlib.use("Agg")
    plt = import_module("matplotlib.pyplot")
    figures = [
        (
            monthly,
            "date",
            "rank_ic",
            "Monthly Rank IC (12-month rolling mean)",
            "rank_ic_timeseries.png",
        ),
        (
            decay_table.groupby(["model", "forward_month"], as_index=False).agg(
                rank_ic=("rank_ic", "mean")
            ),
            "forward_month",
            "rank_ic",
            "Mean Rank IC by forward month",
            "ic_decay.png",
        ),
    ]
    for frame, x, y, title, name in figures:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in frame.groupby("model"):
            group = group.sort_values(x)
            values: pd.Series = group[y]
            if x == "date":
                values = values.rolling(12, min_periods=6).mean()
            ax.plot(group[x], values, label=model)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set(xlabel=x.replace("_", " ").title(), ylabel="Rank IC", title=title)
        if x == "forward_month":
            ax.set_xticks([1, 2, 3])
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)


def evaluate(run_dir: Path, config_path: Path) -> Path:
    load_dotenv()
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    w02_config = Path("configs/week2.yaml")
    data = Path(
        yaml.safe_load(w02_config.read_text(encoding="utf-8"))["output_directory"]
    )
    database = Path("mlruns/tracking.db")
    tracking_uri = "sqlite:///" + database.resolve().as_posix()
    baseline = read_baseline(database)
    sample = load_verified_sample(w02_config, baseline)
    manifest = pd.read_csv(
        run_dir / "walk_forward_folds.csv", dtype={"split_sha256": str}
    )
    if (
        split_manifest_sha256(manifest) != sample.split_sha256
        or baseline.split_sha256 != sample.split_sha256
    ):
        raise ValueError("Evaluation inputs have different split fingerprints.")

    mlflow = import_module("mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    tracked = mlflow.get_run(run_dir.name)
    if tracked.data.tags.get("run_mode") == "smoke":
        raise ValueError("Use the formal Week 3 model run for evaluation.")
    if (
        tracked.info.status != "FINISHED"
        or tracked.data.params.get("split_sha256") != sample.split_sha256
    ):
        raise ValueError("Run directory must belong to the finished Week 3 model run.")

    old = pd.read_csv(
        data / "reports/benchmark_predictions.csv", parse_dates=["date", "label_end"]
    )
    new = pd.read_parquet(run_dir / "predictions.parquet")
    expected = (
        old.loc[old["model"].eq("ols3"), KEYS].sort_values(KEYS).reset_index(drop=True)
    )
    if new.empty or set(old["model"]).intersection(new["model"]):
        raise ValueError(
            "Expected nonempty Week 3 model scores with unique model names."
        )
    for _, group in new.groupby("model"):
        actual = group[KEYS].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    meta = sample.meta.merge(expected, on=KEYS, validate="one_to_one")
    assert len(meta) == len(expected)
    scores = pd.concat([old, new.drop(columns="label_class")], ignore_index=True)
    scores = scores.merge(meta[[*KEYS, "label_class"]], on=KEYS, validate="many_to_one")
    ciks = pd.read_parquet(data / "features_raw.parquet", columns=[*KEYS, "cik"])
    ciks = expected.merge(ciks, on=KEYS, validate="one_to_one", how="left")
    current_sic = industry_snapshot(
        ciks["cik"],
        Path(config["ff12_sic_ranges"]),
        run_dir / "industry",
        os.environ.get("SEC_USER_AGENT", ""),
    )
    industry = ciks.merge(current_sic, on="cik", how="left", validate="many_to_one")
    scores = scores.merge(industry, on=KEYS, how="left", validate="many_to_one")
    scores["industry"] = scores["industry"].fillna("Other")
    scores["sic_available"] = scores["sic_available"].eq(True)

    monthly, decile_table = extended_monthly(scores)
    horizons = {
        length: horizon_labels(meta, data / "prices_development.parquet", length)
        for length in (21, 42, 63)
    }
    decay_table = decay(scores, horizons)
    lags = int(config["newey_west_lags"])
    summary_table = summary(monthly, lags).merge(
        classification(scores), on="model", validate="one_to_one"
    )
    stability_table = stability(
        monthly,
        pd.read_parquet(data / "macro_conditions.parquet"),
        float(config["vix_high_threshold"]),
    )
    coverage_table = (
        industry.assign(sic_available=industry["sic_available"].eq(True))
        .groupby("date", as_index=False)
        .agg(stock_rows=("ticker", "size"), with_valid_sic=("sic_available", "sum"))
    )
    coverage_table["sic_coverage"] = (
        coverage_table["with_valid_sic"] / coverage_table["stock_rows"]
    )

    output = run_dir / "evaluation"
    output.mkdir(exist_ok=True)
    tables = {
        "monthly_metrics": monthly,
        "main_metrics": summary_table,
        "decile_returns": decile_table,
        "paired_newey_west": paired_newey_west(monthly, lags),
        "stability": stability_table,
        "ic_decay": decay_table,
        "sic_coverage": coverage_table,
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    experiment = mlflow.get_experiment(tracked.info.experiment_id).name
    export_trials(tracking_uri, experiment, output / "mlflow_trials.csv")
    chart_tables(monthly, decay_table, output)

    by_row = float(
        coverage_table["with_valid_sic"].sum() / coverage_table["stock_rows"].sum()
    )
    issuer_coverage = current_sic["sic_available"].mean()
    display = summary_table[
        [
            "model",
            "months",
            "ic",
            "rank_ic",
            "icir",
            "nw_t_ic",
            "nw_t_rank_ic",
            "decile_monotonicity",
            "decile_spread",
            "turnover",
            "industry_neutral_ic",
            "sic_coverage",
        ]
    ]
    report = (
        "# Week 3 out-of-sample evaluation\n\n"
        + markdown_table(
            display.rename(columns={"industry_neutral_ic": "industry_neutral_ic [1]"})
        )
        + "\n\n"
        + "[1] "
        + INDUSTRY_NOTE
        + f" Overall stock-month coverage {by_row:.1%}; observed CIK coverage {issuer_coverage:.1%}.\n\n"
        + "Decile monotonicity is the rank correlation of the decile order and mean "
        + "return. Turnover is half the absolute weight change of a gross-one top/bottom "
        + "decile signal. Returns and spreads are gross of costs.\n\n"
        + "IC decay uses non-overlapping excess returns for forward months 1, 2 and 3. "
        + "Month 1 uses the 21-session label. Month 2 uses the close of session 21 "
        + "to the close of session 42, and month 3 uses the close of session 42 to "
        + "the close of session 63. The forecast-date risk-free rate is prorated "
        + "over each window. Windows ending from January 2024 onward are excluded. "
        + "Classification coverage counts actual extreme-tercile "
        + "observations receiving an extreme prediction; AUC uses all labeled observations.\n\n"
        + "Sources: [SEC submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)"
        + " and [Kenneth French Data Library FF12 definitions]"
        + "(https://mba.tuck.dartmouth.edu/pages/faculty/ken.French/Data_Library/det_12_ind_port.html).\n"
    )
    (output / "W03_Evaluation.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/week3_evaluation.yaml")
    )
    args = parser.parse_args()
    print("Week 3 evaluation:", evaluate(args.run_directory, args.config))
