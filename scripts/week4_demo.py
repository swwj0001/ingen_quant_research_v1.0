"""Prepare a five-minute walkthrough using actual Week 2–4 outputs."""

import argparse
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from importlib import import_module
from pathlib import Path

import pandas as pd

from ingen_qr.week4.common import (
    DATABASE,
    TEXT_CONFIG,
    WEEK2,
    WEEK3,
    WEEK3_REPORT_RUN,
    config,
    read_json,
    require_baseline,
)


def benchmark_chart(path: Path, frozen: Path, output: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Week 3 model-zoo predictions are missing: {path}. Supply --week3-run-dir "
            "with the original Week 3 run folder."
        )
    from ingen_qr.evaluation.signal_metrics import monthly_metrics

    model_zoo = monthly_metrics(pd.read_parquet(path))
    bench = pd.read_csv(
        frozen / "reports" / "benchmark_monthly_metrics.csv", parse_dates=["date"]
    )
    table = pd.concat([model_zoo, bench], ignore_index=True).sort_values("date")
    import_module("matplotlib").use("Agg")
    plot = import_module("matplotlib.pyplot")
    fig, ax = plot.subplots(figsize=(10, 5))
    for name, data in table.groupby("model"):
        ax.plot(
            data["date"], data["rank_ic"].rolling(12, min_periods=6).mean(), label=name
        )
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set(
        title="Benchmarks and model zoo",
        ylabel="Rank IC (12-month rolling mean)",
        xlabel="Test month",
    )
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plot.close(fig)


def filing_demo(pilot: Path, output: Path) -> None:
    receipt = read_json(pilot / "completed.json")
    rows = pd.read_parquet(pilot / "firm_month_features.parquet")
    scores = pd.read_parquet(pilot / "filing_scores.parquet")
    from ingen_qr.week4.text_features import FILING_GROUPS

    examples = []
    for family in FILING_GROUPS:
        columns = ["accession", "embedding_score"]
        renamed = {f"{name}_{family}": name for name in columns}
        examples.append(
            rows[["date", "ticker", "cik", "forecast_at", *renamed]]
            .rename(columns=renamed)
            .dropna(subset=["embedding_score"])
            .assign(filing_group=family)
        )
    matched = pd.concat(examples, ignore_index=True).merge(
        scores, on=["cik", "accession"], how="inner", validate="many_to_one"
    )
    if matched.empty:
        raise ValueError("No pilot observation has all three scores to demonstrate.")
    eligible = matched.loc[matched["form"].eq("10-K")]
    row = (
        (eligible if not eligible.empty else matched)
        .sort_values(["date", "ticker"])
        .iloc[-1]
    )
    corpus = pd.read_parquet(
        Path(receipt["issuer_cache"]) / f"{row['cik']}_filings.parquet"
    )
    accession = row["accession"].split(":")[-1]
    record = corpus.loc[corpus["accession"].eq(accession)].iloc[0]
    excerpt = " ".join(record["text"].split())[:1400]
    text = (
        "# SEC filing example\n\n"
        f"{row['ticker']} filing {row['form']} ({accession})\n\n"
        f"- Accepted in New York: {record['accepted_local']}\n"
        f"- First tradable time (UTC): {record['available_at']}\n"
        f"- Monthly forecast cutoff (UTC): {row['forecast_at']}\n"
        f"- Loughran–McDonald tone: {row['lm_tone']:.6f}\n"
        f"- FinBERT tone: {row['finbert_tone']:.6f}\n"
        f"- Embeddings + supervised head output: {row['embedding_score']:.6f}\n\n"
        "The first two are tone scores. Method (c) is a Ridge prediction in next-21-session excess return units.\n\n"
        f"> {excerpt} …\n\n[SEC filing index]({record['index_url']})\n"
    )
    output.write_text(text, "utf-8")


def trial_count() -> tuple[int, int]:
    with closing(
        sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        before, after = db.execute(
            "SELECT SUM(CASE WHEN e.name = 'ingen_qr_week3' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN e.name = 'ingen_qr_week4' THEN 1 ELSE 0 END) "
            "FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id "
            "JOIN tags t ON t.run_uuid = r.run_uuid "
            "WHERE t.key = 'run_kind' AND t.value = 'trial' "
            "AND r.lifecycle_stage = 'active' AND e.name IN ('ingen_qr_week3', 'ingen_qr_week4')"
        ).fetchone()
    return 2 + int(before or 0), int(after or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TEXT_CONFIG)
    parser.add_argument("--week3-run-dir", type=Path)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild thematic development data first.",
    )
    args = parser.parse_args()
    root = Path(config(args.config)["output_directory"])
    gate = require_baseline(root)
    if args.rebuild:
        subprocess.run(
            [sys.executable, "scripts/run_week4_thematic.py", "prepare"], check=True
        )
    directory = root / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    frozen = Path(config(WEEK2)["output_directory"])
    w3 = (
        args.week3_run_dir or Path(config(WEEK3)["output_directory"]) / WEEK3_REPORT_RUN
    )
    benchmark_chart(
        w3 / "predictions.parquet", frozen, directory / "benchmark_model_zoo.png"
    )
    filing_demo(root / "pilot", directory / "filing_example.md")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        check=True,
        capture_output=True,
        text=True,
    )
    count = re.search(r"([0-9]+) tests? collected", proc.stdout)
    if count is None:
        raise RuntimeError("Could not read the collected test count from pytest.")
    w3_trials, w4_trials = trial_count()
    if gate["months"] != 95:
        raise RuntimeError("Baseline months changed.")
    narrative = (
        "# Five-minute mid-point demo\n\n"
        "**0:00–1:10 | Rebuild and data quality.** Show the command "
        "`uv run python scripts/run_week4_thematic.py prepare` and the last rehearsal output. "
        "Open `docs/W02_Data_Quality_Report.md` and `data/week4/thematic/universe_coverage.csv`. "
        "Explain the monthly cutoff and show one missing price or filing and its recorded reason.\n\n"
        "**1:10–2:10 | Model zoo.** Open `benchmark_model_zoo.png`. Show OLS-3, momentum and the "
        f"Week 3 model lines. The RF reproduced across {gate['months']} months, "
        f"with mean Rank IC {gate['mean_rank_ic']:.5f}.\n\n"
        "**2:10–3:35 | One SEC filing.** Open `filing_example.md` and its SEC link. "
        "Show the header time, next-session availability, the extracted passage and all three scores. "
        "Open `data/week4/pilot/head_training_audit.csv` and show the latest labeled observation "
        "used to fit its prediction head.\n\n"
        "**3:35–4:30 | Tests and trials.** Show the repository Actions run for the week4 commit. "
        f"Local pytest collection found {count[1]} tests. For passing status, display "
        "`uv run python scripts/check.py` from the same commit. Show the MLflow trial records, "
        f"with {w3_trials} cumulative Week 1–3 benchmark/trial entries and {w4_trials} Week 4 entries "
        f"({w3_trials + w4_trials} total, including attempted trials).\n\n"
        "**4:30–5:00 | Next decision.** Show `data/week4/thematic/diagnostic.json` and "
        "`data/week4/incremental/H2.txt` if those stages have run. Discuss the Weeks 5–8 scope "
        "and have the reviewer complete and sign the mid-point rubric.\n"
    )
    (directory / "DEMO.md").write_text(narrative, "utf-8")
    print(f"Demo ready: {directory / 'DEMO.md'}")


if __name__ == "__main__":
    main()
