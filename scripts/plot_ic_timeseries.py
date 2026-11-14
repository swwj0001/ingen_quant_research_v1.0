from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

SOURCE = Path("data/week2/reports/benchmark_monthly_metrics.csv")
OUTPUT = Path("docs/figures/W02_IC_Timeseries.png")
PANELS = [("ic", "IC (Pearson)"), ("rank_ic", "Rank IC (Spearman)")]


def main() -> None:
    metrics = pd.read_csv(SOURCE, parse_dates=["date"])
    fig = Figure(figsize=(11, 7))
    axes = fig.subplots(len(PANELS), 1, sharex=True)
    for axis, (column, label) in zip(axes, PANELS, strict=True):
        for model, group in metrics.groupby("model", sort=True):
            series = group.sort_values("date").set_index("date")[column]
            line = axis.plot(series.index, series, alpha=0.35, linewidth=1)[0]
            axis.plot(
                series.index,
                series.rolling(12, min_periods=12).mean(),
                color=line.get_color(),
                linewidth=2,
                label=f"{model} (12-month mean)",
            )
        axis.axhline(0, color="#9ca3af", linewidth=0.7)
        axis.set_ylabel(label)
        axis.legend(loc="upper left")
    axes[0].set_title("W2 benchmarks: monthly out-of-sample IC, 2016-2023")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    summary = metrics.groupby("model")[["ic", "rank_ic"]].agg(["mean", "count"])
    print(summary.round(4).to_string())
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
