# mypy: disable-error-code=import-untyped

"""English reports and static charts generated from recorded development results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{v:.5g}"
                    if isinstance(v, (float, np.floating))
                    else str(v).replace("|", "/")
                )
                for v in row
            )
            + " |"
        )
    return "\n".join(lines)


def coverage_break_section(path: Path) -> str:
    """Include every detected feature/date break in the committable report."""
    breaks = pd.read_csv(path)
    text = "## Feature coverage breaks\n\nA break is an absolute month-to-month coverage change of at least 20 percentage points. The table below contains every row from coverage_breaks.csv.\n\n"
    if breaks.empty:
        return (
            text
            + "No feature coverage changes met the 20-percentage-point threshold.\n\n"
        )
    return text + markdown_table(breaks) + "\n\n"


def create_reports(
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    run_id: str,
    data: Path,
    docs: Path,
) -> list[Path]:
    figures = docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    coverage = quality.pivot(index="feature", columns="date", values="coverage")
    fig, ax = plt.subplots(figsize=(12, max(8, len(coverage) * 0.22)))
    heat = ax.imshow(coverage.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ticks = np.linspace(
        0, len(coverage.columns) - 1, min(10, len(coverage.columns)), dtype=int
    )
    ax.set_xticks(
        ticks,
        [pd.Timestamp(coverage.columns[i]).strftime("%Y-%m") for i in ticks],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(coverage)), coverage.index)
    ax.set_title("Development feature coverage before imputation")
    fig.colorbar(heat, ax=ax, label="Observed fraction")
    fig.tight_layout()
    heat_path = figures / "W02_Feature_Coverage.png"
    fig.savefig(heat_path)
    plt.close(fig)
    exclusions = pd.read_csv(data / "cik_exclusions.csv", parse_dates=["date"])
    monthly = (
        panel.groupby("date")
        .agg(
            research_pool=("ticker", "size"),
            with_forecast_price=("price_at_forecast", "sum"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        exclusions["date"],
        exclusions["original_pool"],
        label="Historical pool",
        color="#6b7280",
    )
    axes[0].plot(
        monthly["date"],
        monthly["research_pool"],
        label="CIK-mapped research pool",
        color="#27658c",
    )
    axes[0].plot(
        monthly["date"],
        monthly["with_forecast_price"],
        label="With forecast-date price",
        color="#bc6c25",
    )
    axes[0].set_ylabel("Stocks")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Universe coverage and the fixed CIK exclusion")
    label_counts = labels.groupby("date")["label_excess_return"].count()
    axes[1].plot(
        label_counts.index, label_counts, label="Mature labels", color="#27658c"
    )
    axes[1].set_ylabel("Stocks")
    axes[1].legend()
    fig.tight_layout()
    pool_path = figures / "W02_Universe_Coverage.png"
    fig.savefig(pool_path)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for model, group in metrics.groupby("model"):
        axes[0].plot(group["date"], group["rank_ic"], label=model, alpha=0.8)
        axes[1].plot(group["date"], group["decile_spread"], label=model, alpha=0.8)
    axes[0].set_title("Development benchmarks: one OOS observation per stock/month")
    axes[0].set_ylabel("Rank IC")
    axes[1].set_ylabel("Top-minus-bottom decile return")
    for ax in axes:
        ax.axhline(0, color="#9ca3af", linewidth=0.7)
        ax.legend()
    fig.tight_layout()
    benchmark_path = figures / "W02_Benchmarks.png"
    fig.savefig(benchmark_path)
    plt.close(fig)
    by_feature = (
        quality.groupby("feature")
        .agg(
            mean_coverage=("coverage", "mean"),
            minimum_coverage=("coverage", "min"),
            outliers=("outliers", "sum"),
            all_missing_months=("all_missing", "sum"),
        )
        .reset_index()
    )
    by_feature.to_csv(data / "feature_quality_summary.csv", index=False)
    status = (
        labels["label_status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="rows")
    )
    staleness = pd.read_csv(data / "macro_staleness.csv")
    age_columns = [str(c) for c in staleness if str(c).endswith("_age_days")]
    stale_summary = pd.DataFrame(
        [
            {
                "series": c.removesuffix("_age_days"),
                "median_age_days": staleness[c].median(),
                "maximum_age_days": staleness[c].max(),
                "share_over_7_days": staleness[c].gt(7).mean(),
            }
            for c in age_columns
        ]
    )
    stale_summary.to_csv(data / "macro_staleness_summary.csv", index=False)
    dq = "# W02 Data Quality Report\n\nResearch only. Development observations: 2010-2023. No sealed-holdout performance was calculated.\n\n"
    dq += "## Eligibility and selection\n\nCompanies without a CIK are excluded uniformly at every date. This uses the supplied mapping snapshot and can still select on later survival; applying the same rule does not remove that bias. No missing-CIK indicator is created. The original universe is preserved in the database.\n\n"
    dq += f"Research rows: {len(panel)}. Mean no-CIK exclusion share: {exclusions['no_cik_exclusion_share'].mean():.2%}. Missing forecast-date prices remain visible in the coverage report. They are excluded from benchmark scoring.\n\n"
    dq += (
        "![Universe coverage](figures/W02_Universe_Coverage.png)\n\n## Feature coverage and outliers\n\n"
        + markdown_table(by_feature)
        + "\n\n![Feature coverage](figures/W02_Feature_Coverage.png)\n\n"
    )
    dq += "The library contains 38 firm features and five conditional interactions, for 43 features. Six raw macro levels are saved separately as conditions and are not cross-sectionally ranked or counted in the feature total. Outlier counts apply only to the 38 raw firm features before 1st/99th-percentile winsorization; interactions are neither winsorized nor ranked again. Firm coverage is measured before imputation. Interaction coverage is the fraction with both the original firm input and macro condition observed, before firm imputation. Entirely missing firm features become neutral zero after transformation; missing macro conditions propagate to their interactions. A month with an entirely unavailable OLS characteristic is not evaluated.\n\n"
    dq += coverage_break_section(data / "coverage_breaks.csv")
    dq += (
        "## Labels\n\n"
        + markdown_table(status)
        + "\n\nMissing terminal prices, including delistings, are not assigned zero return. Their omission can bias the benchmark and must be disclosed. This report does not establish survivorship-free performance.\n\n"
    )
    dq += (
        "## Macro condition coverage and staleness\n\n"
        + markdown_table(
            pd.read_csv(data / "macro_condition_coverage.csv")
            .groupby("condition", as_index=False)
            .agg(
                observed_months=("observed", "sum"),
                total_months=("observed", "size"),
                observed_fraction=("observed", "mean"),
            )
        )
        + "\n\nThe six conditions are stored once per month in macro_conditions.parquet.\n\n"
        + markdown_table(stale_summary)
        + "\n\nThe market-series availability assumption is documented in the feature dictionary and ADR W02-01. Original ALFRED timestamps remain unchanged in the source database.\n\n"
    )
    dq += "## Limitations\n\nCIK mapping is a current-reference coverage screen. Reported shares can be stale and are not free float. Ambiguous concurrent share classes do not receive issuer-shares-based valuation features. Annual-only reporters carry their latest available TTM total, with a 550-day period-end limit. Financial statement availability uses the existing next-session-after-filed-date proxy, not an exact SEC acceptance timestamp. Unapproved ticker aliases are never applied. Source records missing essential prices or corporate actions are not fabricated.\n"
    dq_path = docs / "W02_Data_Quality_Report.md"
    dq_path.write_text(dq, encoding="utf-8")
    report = "# W02 Benchmark Report\n\nResearch only. Untuned OLS-3 and pure momentum; development period only.\n\n"
    report += f"MLflow run ID: `{run_id}`. OOS evaluation starts in 2016; labels crossing 2024 are excluded. The 2015 calibration output is separate.\n\n"
    report += (
        markdown_table(summary)
        + "\n\n![Benchmark metrics](figures/W02_Benchmarks.png)\n\n"
    )
    report += "OLS-3 uses log market capitalization, positive book equity/market capitalization, and 252-minus-21-session momentum. Both models use identical scored stocks/months. Each month the untuned OLS model is refitted on past labels that have matured and passed the one-calendar-month embargo. The 12-month validation block is recorded separately; no hyperparameter is selected. Overlapping validation blocks never create duplicate OOS predictions.\n\n"
    report += "ICIR is mean monthly Rank IC divided by its sample standard deviation, without annualization. Newey-West inference uses six lags and a finite-sample correction. Hit rate is defined only for the return-valued OLS forecast; a rank-only momentum score has no calibrated zero-return threshold. Decile spreads are gross, overlapping 21-session label spreads, not a costed portfolio backtest. Performance remains conditional on the documented CIK/price/fundamental coverage.\n\n"
    report += "The first local experiment run is a benchmark run. No ML model or sealed-holdout evaluation is run by this command. Final hypothesis tests belong to the later one-time holdout evaluation.\n"
    benchmark_report = docs / "W02_Benchmark_Report.md"
    benchmark_report.write_text(report, encoding="utf-8")
    return [dq_path, benchmark_report, heat_path, pool_path, benchmark_path]
