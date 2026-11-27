"""Render Week 4 result summaries from the generated tables."""

from importlib import import_module
from pathlib import Path

import pandas as pd


def table(document, data: pd.DataFrame) -> None:
    grid = document.add_table(rows=1, cols=len(data.columns))
    grid.style = "Light Shading Accent 1"
    for cell, name in zip(grid.rows[0].cells, data.columns, strict=True):
        cell.text = str(name).replace("_", " ")
    for row in data.itertuples(index=False, name=None):
        for cell, value in zip(grid.add_row().cells, row, strict=True):
            cell.text = (
                "—"
                if pd.isna(value)
                else f"{value:.5g}" if isinstance(value, float) else str(value)
            )


def text_report(
    directory: Path, base: Path, results: pd.DataFrame, verdict: str
) -> None:
    doc = import_module("docx").Document()
    doc.add_heading("Week 4 Filing Sentiment", 0)
    doc.add_paragraph("InGen Quant Research | Development sample | Research only")
    doc.add_heading("Incremental rank IC", level=1)
    table(
        doc,
        results[
            [
                "variant",
                "months",
                "mean_rank_ic",
                "paired_months",
                "mean_difference",
                "nw_t",
                "p_two_sided_asymptotic",
            ]
        ],
    )
    doc.add_paragraph(verdict)
    doc.add_paragraph(
        "The two augmented forests and the cached Week 3 forest share the frozen test rows and annual parameters. "
        "Single signals use observed rows and compare against the baseline forest reranked within those same rows. "
        "Tests use six Newey–West lags and the existing small-sample correction. "
        "The all-text and single-signal results are exploratory."
    )
    doc.add_heading("Three sentiment methods", level=1)
    doc.add_paragraph(
        "Loughran–McDonald counts finance-specific words throughout the cleaned document. "
        "FinBERT and the encoder use the same fixed-seed sentence sample, capped at 128 sentences "
        "by default. Method (c) fits separate monthly ridge heads for 10-K/10-Q and 8-K "
        "with matured prior labels and a one-month embargo. The 10-K similarity feature compares "
        "embeddings from consecutive available fiscal-year filings."
    )
    table(
        doc,
        pd.read_csv(base / "method_runtime.csv")[
            ["method", "scoring_seconds", "device", "estimated_compute_cost_usd"]
        ],
    )
    doc.add_heading("Coverage and overlap with existing signals", level=1)
    covered = pd.read_csv(base / "firm_month_coverage.csv")
    covered = covered.groupby("feature")[["observed", "rows"]].sum()
    covered["covered_share"] = covered["observed"] / covered["rows"]
    table(doc, covered[["covered_share"]].reset_index())
    doc.add_paragraph(
        "The latest usable filing can supply a level for up to 180 days under the default configuration. "
        "10-K/10-Q and 8-K supply separate levels and changes. Changes require a new comparable filing "
        "and the H2 forest includes both groups. Each forest keeps the raw missing indicators and "
        "uses the Week 2 monthly winsorization, median imputation and rank transformation. "
        "Filing-level timestamp discrepancies "
        "and exclusion reasons are in the accompanying coverage tables."
    )
    partial = (
        pd.read_csv(directory / "partial_correlations.csv")
        .groupby("feature")[
            ["text_return_partial_rank", "text_short_momentum_partial_rank"]
        ]
        .mean()
        .reset_index()
    )
    table(doc, partial)
    doc.save(directory / "W04_Text_Signal_Report.docx")
