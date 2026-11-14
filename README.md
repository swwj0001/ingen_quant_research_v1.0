\## Environment



Python 3.11.18. Dependencies are managed with uv and locked in uv.lock.



\## Installation



```bash

uv sync --locked

```



\## Project Layout



\- src/ingen\_qr/: data, features, models, backtesting, portfolio construction and evaluation

\- configs/: configuration

\- tests/: tests

\- docs/: documentation

\- scripts/: runnable scripts

\- notebooks/: exploratory analysis

\- weekly/: weekly logs

\- data/: raw data, processed data and caches



\## Rules



\- Each record includes an available\_at timestamp. Predictions use only information available at the prediction time.

\- notebooks/: exploration only. No result may depend on a notebook.



## Pipeline commands

Fill FRED_API_KEY and SEC_USER_AGENT in .env.

Run all code checks:

```bash
uv run python scripts/check.py
```

Check three stocks and all data sources:

```bash
uv run python -m ingen_qr.pipeline --smoke
```

Rebuild the full configured raw dataset and SQL panel:

```bash
uv run python -m ingen_qr.pipeline
```

See [pipeline documentation](docs/PIPELINE.md) for time rules, data coverage,
SQL examples and the difference between smoke and full builds.
