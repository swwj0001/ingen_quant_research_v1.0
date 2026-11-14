# mypy: disable-error-code=import-untyped

"""One-command W2 features, labels, benchmarks, tracked results and reports."""

import argparse
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from ingen_qr.evaluation.benchmarks import run_benchmarks
from ingen_qr.evaluation.labels import make_labels
from ingen_qr.evaluation.reporting import create_reports
from ingen_qr.features.macro import FEATURES as MACRO_FEATURES
from ingen_qr.features.panel import (
    FEATURES,
    FIRM_FEATURES,
    build_monthly_panel,
    save_parquet,
)
from ingen_qr.features.preprocessing import preprocess_panel


def run(config_path: Path, allow_network: bool = False) -> dict[str, object]:
    # Set this before importing the SDK, which initializes its telemetry client.
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    mlflow = import_module("mlflow")
    config = yaml.safe_load(config_path.read_text("utf-8"))
    data = Path(config["output_directory"])
    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    with duckdb.connect(str(config["database"]), read_only=True) as connection:
        panel, cohorts, market = build_monthly_panel(
            connection, config, reports, allow_network
        )
    save_parquet(panel.drop(columns=MACRO_FEATURES), data / "features_raw.parquet")
    conditions = pd.read_parquet(data / "macro_conditions.parquet")
    processed, quality = preprocess_panel(panel, FIRM_FEATURES, conditions)
    save_parquet(processed, data / "features_processed.parquet")
    quality.to_csv(reports / "feature_quality.csv", index=False)
    prices = pd.read_parquet(data / "prices_development.parquet")
    labels = make_labels(cohorts, prices, market, str(config["development_end"]))
    save_parquet(labels, data / "labels.parquet")
    labels["label_status"].value_counts().rename_axis("status").reset_index(
        name="rows"
    ).to_csv(reports / "label_diagnostics.csv", index=False)
    changes = quality.sort_values(["feature", "date"]).copy()
    changes["coverage_change"] = changes.groupby("feature")["coverage"].diff()
    changes.loc[changes["coverage_change"].abs() >= 0.20].to_csv(
        reports / "coverage_breaks.csv", index=False
    )
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(
        "sqlite:///" + (Path("mlruns") / "tracking.db").resolve().as_posix()
    )
    mlflow.set_experiment("ingen_qr_week2")
    existing = mlflow.search_runs(
        experiment_names=["ingen_qr_week2"], order_by=["attributes.start_time ASC"]
    )
    if not existing.empty and existing.iloc[0].get("tags.run_kind", "") != "benchmark":
        raise ValueError(
            "The first experiment run was not a benchmark. Preserve the tracking history and resolve this explicitly."
        )
    with mlflow.start_run(
        run_name="baseline_ols3_momentum",
        tags={
            "run_kind": "benchmark",
            "research_only": "true",
            "holdout_locked": "true",
        },
    ) as active:
        run_id = active.info.run_id
        mlflow.log_params(
            {
                "development_start": config["development_start"],
                "development_end": config["development_end"],
                "feature_count": len(FEATURES),
                "firm_feature_count": len(FIRM_FEATURES),
                "interaction_count": len(FEATURES) - len(FIRM_FEATURES),
                "macro_condition_count": len(MACRO_FEATURES),
                "seed": 1,
                "nw_lags": 6,
                "embargo_months": 1,
                "validation_months": 12,
                "macro_rule": "first_vintage_next_XNYS_open",
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        )
        for name in (
            "features_raw",
            "features_processed",
            "macro_conditions",
            "labels",
        ):
            with (data / (name + ".parquet")).open("rb") as source:
                mlflow.log_param(
                    name + "_sha256", hashlib.file_digest(source, "sha256").hexdigest()
                )
        print("Running the OLS-3 and momentum benchmarks.", flush=True)
        _, metrics, summary = run_benchmarks(panel, processed, labels, config, reports)
        for row in summary.to_dict("records"):
            for metric_name, value in row.items():
                if (
                    metric_name != "model"
                    and pd.notna(value)
                    and np.isfinite(float(value))
                ):
                    mlflow.log_metric(
                        str(row["model"]) + "." + str(metric_name), float(value)
                    )
        paths = create_reports(
            panel, quality, labels, metrics, summary, run_id, reports, docs
        )
        for path in paths:
            mlflow.log_artifact(str(path), artifact_path="reports")
        mlflow.log_artifact(str(config_path), artifact_path="config")
    dictionary = docs / "W02_Feature_Dictionary.md"
    if dictionary.exists():
        text = dictionary.read_text("utf-8").split("\n<!-- GENERATED COVERAGE -->")[0]
        coverage = quality.groupby("feature")["coverage"].mean()
        text += "\n<!-- GENERATED COVERAGE -->\n\n## Development coverage before imputation\n\n| Feature | Mean monthly coverage |\n| --- | --- |\n"
        text += (
            "\n".join(
                f"| {name} | {coverage.get(name, np.nan):.2%} |" for name in FEATURES
            )
            + "\n"
        )
        condition_coverage = conditions[MACRO_FEATURES].notna().mean()
        text += "\n## Macro condition coverage (not counted among the 43 features)\n\n| Condition | Observed monthly fraction |\n| --- | --- |\n"
        text += (
            "\n".join(
                f"| {name} | {condition_coverage[name]:.2%} |"
                for name in MACRO_FEATURES
            )
            + "\n"
        )
        dictionary.write_text(text, encoding="utf-8")
    result: dict[str, object] = {
        "status": "development_completed",
        "feature_count": len(FEATURES),
        "firm_feature_count": len(FIRM_FEATURES),
        "interaction_count": len(FEATURES) - len(FIRM_FEATURES),
        "macro_condition_count": len(MACRO_FEATURES),
        "panel_rows": len(panel),
        "mlflow_run_id": run_id,
        "holdout_opened": False,
        "reports": [str(p) for p in paths],
    }
    (reports / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild W2 from the existing database; holdout remains locked."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/week2.yaml"))
    parser.add_argument(
        "--repair-prices",
        action="store_true",
        help="Also retry bounded development requests for non-timezone price failures.",
    )
    args = parser.parse_args()
    run(args.config, args.repair_prices)


if __name__ == "__main__":
    main()
