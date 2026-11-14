# ingen_quant_research_v1.0

This is my research project on predicting the next 21 trading days of excess returns for historical S&P 500 stocks. I have added the data loaders, features, return labels, validation code, and the OLS-3 and momentum benchmarks.

I use Python 3.11.18 and uv. Run these commands from the project folder.

## Running it

Install the dependencies first.

```bash
uv sync --locked
```

Fill in `FRED_API_KEY` and `SEC_USER_AGENT` in `.env` before downloading data. The SEC user agent needs a real contact email.

To check the code and run the tests, use

```bash
uv run python scripts/check.py
```

This runs Ruff, Black, mypy and pytest.

To rebuild the full configured source dataset and DuckDB views, use

```bash
uv run python -m ingen_qr.pipeline
```

The database is `data/ingen_qr.duckdb`. Successful downloads are cached for reuse. The source settings are in `configs/pipeline.yaml`.

There is also a smaller build for checking the data sources.

```bash
uv run python -m ingen_qr.pipeline --smoke
```

Smoke mode replaces the processed panels with a small stock sample plus SPY. Run the full build again before W2. Keep these panels unchanged while W2 is running. More details are in [docs/PIPELINE.md](docs/PIPELINE.md).

Once the full database is ready, prepare the output rules and run W2.

```bash
uv run python scripts/prepare_week2.py
uv run python scripts/run_week2.py
```

The preparation script adds `mlruns/` and `.hypothesis/` to `.gitignore`. The existing `/data/` rule covers the root data folder.

The W2 command generates features, labels, benchmark results and report figures using `configs/week2.yaml`. It prints progress by month, then a summary and an MLflow run ID.

To also retry some failed price downloads for the development period, use

```bash
uv run python scripts/run_week2.py --repair-prices
```

Either mode can reuse existing raw price files. The retry option does not recover every missing stock. Remaining failures are listed in `data/week2/reports/price_recovery.csv`. Proposed ticker renames need to be checked before use.

## Week 2

There are 38 firm characteristics and five macro interactions, giving 43 features. Six macro conditions are saved separately. I normalize the firm characteristics first, then multiply them by the raw macro values. The products are not ranked again, so differences between months are kept.

OLS-3 uses size, book-to-market and momentum. The pure momentum benchmark uses the same stocks and months. Development uses 2010-2023, with 2015 for calibration and monthly out-of-sample evaluation from 2016. Labels ending in 2024 are left out. The 2024-2026 holdout stays locked.

The main files under `data/week2/` are `features_raw.parquet`, `features_processed.parquet`, `macro_conditions.parquet` and `labels.parquet`. Metrics and coverage checks go in `data/week2/reports/`, and MLflow records go in `mlruns/`.

The script also writes `docs/W02_Benchmark_Report.md` and `docs/W02_Data_Quality_Report.md`, saves charts in `docs/figures/`, and updates coverage in the [feature dictionary](docs/W02_Feature_Dictionary.md). The full-data results still need to be checked before acceptance.

## A few notes

Inputs have an `available_at` timestamp. Features only use information available by the forecast time. Daily returns follow XNYS sessions without filling or crossing missing prices. SPY is the market proxy for Beta, while Fama-French factors are kept for later attribution.

Companies without a CIK mapping are excluded, and the monthly exclusion rates are reported. A current CIK list can miss companies that later disappeared or changed names, so this can still affect the results. Missing terminal prices stay missing rather than becoming zero returns.

For DGS3MO, DGS10, DBAA and VIXCLS, I use the earliest non-missing archived value and assume it was available at the next XNYS session open after the observation date. The original ALFRED timestamps are kept. This assumption needs supervisor confirmation; see [ADR W02-01](docs/adr/W02-01-market-series-availability.md).

The data comes from historical constituent records, Yahoo Finance through yfinance, SEC EDGAR, FRED/ALFRED and the Kenneth French Data Library. Provider terms and attribution still apply.

Code is in `src/ingen_qr/`, settings in `configs/`, tests in `tests/`, research notes in `docs/`, and weekly logs in `weekly/`. Notebooks are for exploration; reported results should come from scripts.

Keep `.env`, downloaded data, caches and `mlruns/` out of Git. This project is for research use only.
