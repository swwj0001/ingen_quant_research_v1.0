"""Diagnose thematic transfer and run the single remedy authorized by the rule."""

import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ingen_qr.evaluation.signal_metrics import monthly_metrics
from ingen_qr.evaluation.splitters import training_indices
from ingen_qr.evaluation.week3_samples import file_sha256
from ingen_qr.models.week3 import Regressor
from ingen_qr.week4 import baseline
from ingen_qr.week4.common import (
    KEYS,
    export_trials,
    log_metrics,
    parquet,
    read_json,
    trial,
    write_json,
)
from ingen_qr.week4.paired import paired_monthly
from ingen_qr.week4.reports import table

TRANSFER = "sp500_rf_transfer"
THEME_ONLY = "thematic_only_rf"


def choice(values: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str]:
    checks = [
        ("thin_month_share", settings["thin_month_max_share"], "too many thin months"),
        (
            "short_history_share",
            settings["short_history_max_share"],
            "too many short histories",
        ),
        (
            "fundamental_coverage",
            settings["fundamental_minimum_coverage"],
            "low fundamental coverage",
        ),
    ]
    failed = []
    for key, threshold, reason in checks:
        value = values[key]
        if value is None or not np.isfinite(value):
            failed.append(f"{key} unavailable")
        elif value < threshold if key == "fundamental_coverage" else value > threshold:
            failed.append(reason)
    if failed:
        return "pooled", "Structural checks failed: " + ", ".join(failed) + "."
    return "shrinkage", "All three structural checks passed."


def forecast_plan(sp, meta: pd.DataFrame, minimum: int, remedy: str):
    months = meta["date"].dt.to_period("M")
    for fold in sp.folds:
        test = meta.loc[months.eq(fold.test_month.to_period("M"))]
        if test.empty:
            continue
        cutoff = pd.Timestamp(test["forecast_at"].min())
        idx = training_indices(meta, fold.test_month, cutoff, embargo_months=1)
        base_idx = np.empty(0, dtype=int)
        if remedy == "pooled":
            base = sp.meta.iloc[fold.refit]
            theme_keys = pd.MultiIndex.from_frame(meta.iloc[idx][KEYS])
            base_idx = fold.refit[
                ~pd.MultiIndex.from_frame(base[KEYS]).isin(theme_keys)
            ]
        if len(idx) + len(base_idx) >= minimum:
            yield fold, test, idx, base_idx


def transfer(
    panels: pd.DataFrame, matrix: pd.DataFrame, directory: Path, signature: str
) -> pd.DataFrame:
    joblib = import_module("joblib")
    months = panels["date"].dt.strftime("%Y-%m")
    records = []
    print(
        "Expected thematic transfer fits: 0 (reuse cached Week 3 baseline models).",
        flush=True,
    )
    for month, block in panels.loc[months.isin(baseline.MONTHS)].groupby(months):
        path = directory / signature / f"{month}.joblib"
        if not path.is_file():
            raise RuntimeError(
                f"Missing cached Week 3 RF for {month}. Run the baseline command again."
            )
        model = joblib.load(path)
        records.append(
            block[baseline.PREDICTION_COLUMNS].assign(
                model=TRANSFER, prediction=model.predict(matrix.loc[block.index])
            )
        )
    if not records:
        raise ValueError("No comparable thematic and S&P 500 test months.")
    return pd.concat(records, ignore_index=True)


def thematic_prediction(
    remedy: str,
    sp,
    meta: pd.DataFrame,
    matrix: pd.DataFrame,
    transferred: pd.DataFrame,
    settings: dict[str, Any],
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = list(forecast_plan(sp, meta, int(settings["minimum_training_rows"]), remedy))
    print(
        f"Expected thematic {remedy} fits: {len(plan)} (fixed Week 3 RF settings, no tuning).",
        flush=True,
    )
    if not plan:
        raise ValueError("No thematic fold has the minimum mature training rows.")
    options = baseline.annual_parameters()
    results, audit = [], []
    for fold, test, idx, base_idx in plan:
        train = meta.iloc[idx]
        train_x = matrix.iloc[idx].copy()
        test_x = matrix.loc[test.index].copy()
        x = train_x
        y = train["label_excess_return"]
        if remedy == "pooled":
            sp_x = sp.matrix.iloc[base_idx].assign(is_thematic=0.0)
            x = pd.concat([sp_x, train_x.assign(is_thematic=1.0)], ignore_index=True)
            y = pd.concat(
                [sp.meta.iloc[base_idx]["label_excess_return"], y], ignore_index=True
            )
            test_x["is_thematic"] = 1.0
        model = Regressor("random_forest", options[fold.test_month.year], 1)
        model.fit(x, y, train)
        values = model.predict(test_x)
        rows = test[baseline.PREDICTION_COLUMNS].assign(model=remedy, prediction=values)
        weight = None
        if remedy == "shrinkage":
            reference = rows.merge(
                transferred[[*KEYS, "prediction"]],
                on=KEYS,
                validate="one_to_one",
                suffixes=("", "_transfer"),
            )
            weight = len(train) / (len(train) + float(settings["shrink_prior_rows"]))
            rows["prediction"] = (
                weight * reference["prediction"].rank(pct=True).to_numpy()
                + (1 - weight)
                * reference["prediction_transfer"].rank(pct=True).to_numpy()
            )
            rows["thematic_only_prediction"] = values
        audit.append(
            {
                "test_month": fold.test_month,
                "theme_training_rows": len(train),
                "pooled_sp500_training_rows": len(base_idx),
                "latest_thematic_label_end": train["label_end"].max(),
                "latest_thematic_label_available_at": train["label_available_at"].max(),
                "forecast_at": test["forecast_at"].min(),
                "thematic_weight": weight,
            }
        )
        results.append(rows)
        print(f"Thematic {remedy}: {fold.test_month:%Y-%m}", flush=True)
    predictions = pd.concat(results, ignore_index=True)
    audit_frame = pd.DataFrame(audit)
    parquet(predictions, directory / "remedy_predictions.parquet")
    audit_frame.to_csv(directory / "remedy_fold_audit.csv", index=False)
    return predictions, audit_frame


def remedy_comparison(
    predictions: pd.DataFrame, transferred: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    before = predictions[KEYS].merge(transferred, on=KEYS, validate="one_to_one")
    if len(before) != len(predictions):
        raise ValueError(
            "The before and after remedy scores must cover the same stock-months."
        )
    before_ic = monthly_metrics(before).set_index("date")["rank_ic"]
    after_ic = monthly_metrics(predictions).set_index("date")["rank_ic"]
    paired = (
        pd.concat(
            [before_ic.rename("before_rank_ic"), after_ic.rename("after_rank_ic")],
            axis=1,
            join="inner",
        )
        .dropna()
        .sort_index()
    )
    paired["rank_ic_change"] = paired["after_rank_ic"] - paired["before_rank_ic"]
    return {
        "before_mean_rank_ic": paired["before_rank_ic"].mean(),
        "after_mean_rank_ic": paired["after_rank_ic"].mean(),
        **paired_monthly(after_ic, before_ic),
    }, paired


def matching_size(
    sp: pd.DataFrame, theme: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = theme.groupby("date").size()
    rows = []
    for date, group in sp.groupby("date"):
        size = int(sizes.get(date, 0))
        if not 10 <= size <= len(group):
            continue
        values = []
        for _ in range(reps):
            subset = group.iloc[rng.choice(len(group), size, replace=False)]
            rank_ic = (
                subset["prediction"].rank().corr(subset["label_excess_return"].rank())
            )
            if np.isfinite(rank_ic):
                values.append(rank_ic)
        rows.append(
            {
                "date": date,
                "thematic_stocks": size,
                "subsamples": len(values),
                "mean_rank_ic": float(np.mean(values)) if values else np.nan,
                "rank_ic_sd": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["date", "thematic_stocks", "subsamples", "mean_rank_ic", "rank_ic_sd"],
    )


def thematic_report(
    directory: Path,
    coverage: pd.DataFrame,
    diagnostics: dict[str, Any],
    result: dict[str, Any] | None,
) -> None:
    document = import_module("docx").Document()
    document.add_heading("Week 4 Thematic Universe Gap Analysis", 0)
    document.add_paragraph("InGen Quant Research | Development sample | Research only")
    document.add_heading("Universe and data", level=1)
    counts = (
        coverage.groupby("category")
        .agg(
            candidates=("ticker", "size"),
            eligibility=("modeled_security_type", "sum"),
            stock_months=("scored_rows", "sum"),
        )
        .reset_index()
    )
    table(document, counts)
    document.add_paragraph(
        "Candidate businesses serve robotics, manufacturing automation, motion control, machine vision, "
        "edge computing or industrial AI software. US exchange-listed stocks and ADRs are eligible. "
        "U.S.-listed foreign common shares such as STM retain their own security type. "
        "OTC and foreign-exchange listings appear in the coverage table. Some listing starts use the first "
        "observed vendor quote. The research list is curated retrospectively and can omit delisted companies."
    )
    document.add_heading("What generalises", level=1)
    summary = pd.DataFrame(
        [
            (name, diagnostics[name])
            for name in (
                "paired_months",
                "sp500_rank_ic_matched",
                "thematic_rank_ic_matched",
                "mean_difference",
                "nw_t",
                "thin_month_share",
                "short_history_share",
                "fundamental_coverage",
                "thematic_return_volatility",
                "sp500_return_volatility",
                "size_matched_sp500_ic_sd",
                "thematic_rank_ic_sd",
                "ticker_month_overlap_share",
            )
        ],
        columns=["diagnostic", "value"],
    )
    table(document, summary)
    document.add_heading("Decision and evaluated change", level=1)
    document.add_paragraph(
        f"Rule-selected setting {diagnostics['selected_remedy']}. {diagnostics['decision_reason']}"
    )
    if result:
        table(document, pd.DataFrame([result]).drop(columns="variant"))
    else:
        document.add_paragraph(
            "The gap diagnosis is shown above. The current report has no completed remedy test."
        )
    document.add_paragraph(
        "Rank IC is evaluated on the development sample. Remedies reuse fixed Week 3 settings. "
        "Calendar-aligned comparisons may share issuers with the S&P 500. Portfolio costs and sealed "
        "holdout tests come later in the internship plan."
    )
    document.save(directory / "W04_Thematic_Universe_Gap_Analysis.docx")


def run(
    settings: dict[str, Any], root: Path, gate: dict[str, Any], apply_remedy: bool
) -> None:
    directory = root / "thematic"
    receipt_path = directory / "panel_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("Run thematic prepare first.")
    receipt = read_json(receipt_path)
    if (
        receipt["baseline_signature"] != gate["signature"]
        or receipt["short_history_sessions"] != settings["short_history_sessions"]
        or receipt["candidates_sha256"] != file_sha256(Path(settings["candidates"]))
        or any(
            file_sha256(directory / f"{name}.parquet") != digest
            for name, digest in receipt["input_sha256"].items()
        )
    ):
        raise ValueError("Thematic input fingerprints differ. Re-run thematic prepare.")
    meta = pd.read_parquet(directory / "meta.parquet")
    matrix = pd.read_parquet(directory / "matrix.parquet")
    sp = baseline.sample()
    if meta["date"].ge("2024-01-01").any() or meta["label_end"].ge("2024-01-01").any():
        raise PermissionError("The thematic matrix contains sealed holdout rows.")
    pd.testing.assert_frame_equal(meta[KEYS], matrix[KEYS])
    matrix = matrix.loc[:, sp.matrix.columns]
    frozen = pd.read_parquet(root / "baseline" / "predictions.parquet")
    theme_score = transfer(meta, matrix, root / "baseline", gate["signature"])
    parquet(theme_score, directory / "transfer_predictions.parquet")
    metrics = monthly_metrics(theme_score)
    sp_metrics = monthly_metrics(frozen)
    sp_ic, theme_ic = (
        sp_metrics.set_index("date")["rank_ic"],
        metrics.set_index("date")["rank_ic"],
    )
    gap = paired_monthly(sp_ic, theme_ic)
    common = sp_ic.index.intersection(theme_ic.index)
    joint = pd.concat(
        [sp_ic.rename("sp500_rank_ic"), theme_ic.rename("thematic_rank_ic")], axis=1
    )
    joint.to_csv(directory / "transfer_monthly_rank_ic.csv")
    matched_size = matching_size(
        frozen, theme_score, int(settings["size_match_repeats"]), int(settings["seed"])
    )
    matched_size.to_csv(directory / "size_matched_rank_ic.csv", index=False)
    panel = pd.read_parquet(
        directory / "features_raw.parquet",
        columns=[*KEYS, "history_sessions", "fundamental_coverage"],
    )
    periods = sorted(frozen["date"].unique())
    sizes = theme_score.groupby("date").size().reindex(periods, fill_value=0)
    coverage = pd.read_csv(directory / "universe_coverage.csv")
    test_panel = panel.loc[panel["date"].isin(periods)]
    overlap = theme_score[KEYS].merge(frozen[KEYS], on=KEYS, validate="one_to_one")
    diagnostics = {
        **gap,
        "sp500_rank_ic_matched": sp_ic.loc[common].mean(),
        "thematic_rank_ic_matched": theme_ic.loc[common].mean(),
        "thin_month_share": sizes.lt(int(settings["thin_month_minimum_stocks"])).mean(),
        "short_history_share": test_panel["history_sessions"]
        .lt(settings["short_history_sessions"])
        .mean(),
        "fundamental_coverage": test_panel["fundamental_coverage"].mean(),
        "median_scored_stocks": sizes.median(),
        "thematic_return_volatility": theme_score.loc[
            theme_score["date"].isin(common), "label_excess_return"
        ].std(),
        "sp500_return_volatility": frozen.loc[
            frozen["date"].isin(common), "label_excess_return"
        ].std(),
        "size_matched_sp500_ic_sd": matched_size["rank_ic_sd"].mean(),
        "thematic_rank_ic_sd": theme_ic.loc[common].std(),
        "ticker_month_overlap_share": len(overlap) / len(theme_score),
    }
    diagnostics = {
        key: (
            None
            if not np.isfinite(value)
            else int(value) if key == "paired_months" else float(value)
        )
        for key, value in diagnostics.items()
    }
    selected, reason = choice(diagnostics, settings)
    diagnostics.update(selected_remedy=selected, decision_reason=reason)
    write_json(diagnostics, directory / "diagnostic.json")
    print(f"Thematic diagnosis selected {selected}: {reason}", flush=True)
    result = None
    if apply_remedy:
        identity = {
            "baseline": gate["signature"],
            "panel": receipt["input_sha256"],
            "remedy": selected,
            "minimum_rows": settings["minimum_training_rows"],
            "shrink_prior_rows": settings["shrink_prior_rows"],
            "code": file_sha256(Path(__file__)),
        }
        signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        cache = directory / "remedies" / signature
        result_path = cache / "remedy_predictions.parquet"
        finished = cache / "completed.json"
        if finished.is_file() and read_json(finished)["sha256"] == file_sha256(
            result_path
        ):
            print(f"Expected thematic {selected} fits: 0 (verified cache).", flush=True)
            predictions = pd.read_parquet(result_path)
        else:
            cache.mkdir(parents=True, exist_ok=True)
            planned = len(
                list(
                    forecast_plan(
                        sp, meta, int(settings["minimum_training_rows"]), selected
                    )
                )
            )
            with trial(
                "thematic_" + selected,
                {
                    "split_sha256": baseline.SPLIT,
                    "planned_fits": planned,
                    "remedy": selected,
                },
            ) as mlflow:
                predictions, _ = thematic_prediction(
                    selected,
                    sp,
                    meta,
                    matrix,
                    theme_score,
                    settings,
                    cache,
                )
                forward, _ = remedy_comparison(predictions, theme_score)
                log_metrics(mlflow, forward)
            write_json({"sha256": file_sha256(result_path)}, finished)
        forward, compared = remedy_comparison(predictions, theme_score)
        result = {"variant": selected, **forward}
        pd.DataFrame([result]).to_csv(directory / "remedy_evaluation.csv", index=False)
        compared.to_csv(directory / "remedy_monthly_comparison.csv")
        print(
            f"{selected}: {result['paired_months']} paired months, mean Rank IC "
            f"{result['before_mean_rank_ic']:.5f} before and "
            f"{result['after_mean_rank_ic']:.5f} after.",
            flush=True,
        )
        if selected == "shrinkage":
            standalone = predictions.assign(
                model=THEME_ONLY, prediction=predictions["thematic_only_prediction"]
            )
            monthly_metrics(standalone).to_csv(
                directory / "thematic_only_monthly_rank_ic.csv", index=False
            )
    thematic_report(directory, coverage, diagnostics, result)
    if apply_remedy:
        export_trials(root / "W04_Experiment_Log.csv")
