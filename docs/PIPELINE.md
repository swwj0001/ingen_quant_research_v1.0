# W01 data pipeline

Python 3.11.18. Dependency manager: uv. The existing pyproject.toml and uv.lock
remain the dependency specification. Run `uv sync --locked` after cloning.

## Commands

Fill `.env` using `.env.example`. Get a free FRED key at
https://fred.stlouisfed.org/docs/api/api_key.html.
SEC_USER_AGENT must identify this project and contain a real contact email.

```bash
uv run python scripts/check.py
uv run python -m ingen_qr.pipeline --smoke
uv run python -m ingen_qr.pipeline
```

The last command rebuilds the complete configured raw dataset from an empty
`data/` directory before modelling. It downloads all stocks in the historical
monthly cohorts, the factor files, four FRED series and mapped SEC company facts.
The smoke command uses AAPL, MSFT and IBM and is not a full build.
`--refresh` requests new downloads. Repeating without it reuses the disk cache.
The network must permit access to Yahoo, FRED and SEC.

## Data and time rules

- Prices: preserve source prices, dividends and splits. Reconstruct native
  prices and a forward total-return index with base 100. Read the full source
  history through retrieval time to undo later split adjustments. The model
  panel ends on 2026-09-30. Availability is assumed to be exchange close plus
  30 minutes. Incomplete sessions are excluded.
- FRED: DGS3MO, T10Y3M, BAA10Y and VIXCLS. Request ALFRED real-time intervals,
  retain revisions, and preserve missing observations. Availability is midnight
  New York time following the later of observation date and vintage start date.
  DGS3MO is retained in its original annual-percent units. No one-month French
  RF values are substituted for it.
- SEC: preserve accession number, filing date, period start/end, form, fiscal
  period, tag, namespace, unit and value. Preserve separate restatements.
  Availability uses the next trading-session open after the filing date, a
  conservative date-only rule. Download complete companyfacts JSON files and
  normalize the financial tags listed in the configuration.
- French factors: retain the captured source version and download timestamp.
  Use them for ex-post attribution, not as historical prediction inputs. Return
  units are decimals. `ff_rf` is the daily return derived from the one-month
  T-bill series in the source file.
- Membership: use the pinned community history from fja05680/sp500. Its commit
  is a source version, not a historical announcement timestamp. Effective
  session open is a conservative knowledge-time proxy, not an announcement time.
  Monthly forecasts are formed one hour after the prior session closes, before
  the next month's first opening. Select stocks belonging to the index at that
  forecast time. The pinned source is dated 2026-09-07, sufficient for the final
  August 31 forecast for September. Do not extrapolate it to later forecasts.

Two membership additions are checked against S&P announcements:

- TSLA, effective 2020-12-21:
  https://press.spglobal.com/2020-11-16-Tesla-Set-to-Join-S-P-500
- GEHC, effective 2023-01-04:
  https://press.spglobal.com/2022-12-28-GE-HealthCare-Technologies-Set-to-Join-S-P-500-Vornado-Realty-Trust-to-Join-S-P-MidCap-400-RXO-to-Join-S-P-SmallCap-600

These checks do not establish complete historical coverage. Current SEC ticker
mapping can miss former tickers or confuse reused symbols. Free Yahoo histories
can miss delisted stocks.
Price, CIK and monthly coverage reports retain these gaps. No missing return or
financial value is filled with zero. Review these reports before modelling.

## Storage and SQL

Raw downloads: `data/raw/`. Disk cache: `data/cache/`.
Processed tables use monthly date partitions:
`data/processed/<table>/partition_date=YYYY-MM-01/*.parquet`.
All five tables include `available_at`.
The local database is `data/ingen_qr.duckdb`.

```python
import duckdb
from ingen_qr.data.storage import as_of

with duckdb.connect("data/ingen_qr.duckdb", read_only=True) as connection:
    panel = as_of(connection, "fred", "2016-01-04T12:00:00Z")
```

`as_of` excludes unavailable records before selecting the latest eligible
revision. Query the universe by its explicit cohort date and forecast time.
Reports are in `data/reports/`. A successful full run writes
`build_report.json` with `mode=full`. A smoke report is not evidence of a full
download. Data, caches, .env and the virtual environment are excluded from Git.
Notebooks are for exploration only. Reported results must come from scripts.

## Validation and CI

`scripts/check.py` runs ruff, black, mypy and pytest together. Tests use offline
fixtures, including future split invariance, early closes, FRED vintage leakage,
SEC filing delay and restatements, historical membership and an entire small
build producing all five SQL tables. They require no credentials or network.
GitHub Actions runs the same checks on every push and pull request. Its actual
remote run is verified after uploading the repository.

Source documentation:

- https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/about/developer-resources

The common Downloader supplies throttling, exponential-backoff retries and disk
caching across sources. Keep the research-design decisions and sealed evaluation
period in the research brief. This package does not train or tune any models.
