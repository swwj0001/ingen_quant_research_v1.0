import hashlib
import os
from importlib import import_module
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
import yaml

from ingen_qr.data.storage import ParquetStore
from ingen_qr.features.macro import FEATURES as MACRO_FEATURES
from ingen_qr.features.macro import INTERACTIONS
from ingen_qr.features.panel import FEATURES, FIRM_FEATURES
from ingen_qr.week2 import run


def build_fixture(root: Path) -> Path:
    calendar = xcals.get_calendar("XNYS", start="2008-01-01", end="2011-08-31")
    dates = calendar.sessions_in_range("2008-01-02", "2011-06-30")
    prices, facts, cohorts, fred = [], [], [], []
    random = np.random.default_rng(42)
    for i in range(13):
        ticker = f"S{i:02d}" if i < 12 else "SPY"
        close = 50 * np.cumprod(
            1 + random.normal(0.0002 + i * 0.00001, 0.008, len(dates))
        )
        for j, day in enumerate(dates):
            prices.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "available_at": calendar.session_close(day)
                    + pd.Timedelta(minutes=30),
                    "raw_close": close[j],
                    "raw_open": close[j] * 0.999,
                    "adjusted_open": close[j] * 0.999,
                    "adjusted_close": close[j],
                    "volume": 100000.0 + i * 2000,
                    "split_ratio": 1.0,
                }
            )
        if i == 12:
            continue
        for year in [2007, 2008, 2009, 2010]:
            values = {
                "Assets": 1000 + i * 30 + (year - 2007) * 50,
                "StockholdersEquity": 500 + i * 10,
                "Liabilities": 500 + i * 20,
                "NetIncomeLoss": 100 + i,
                "Revenues": 1500 + i * 30,
                "GrossProfit": 600 + i * 20,
                "NetCashProvidedByUsedInOperatingActivities": 140 + i,
                "PaymentsToAcquirePropertyPlantAndEquipment": 50 + i,
                "CommonStockSharesOutstanding": 100 + i,
            }
            for tag, value in values.items():
                if tag == "GrossProfit" and year < 2009:
                    continue
                instant = tag in [
                    "Assets",
                    "StockholdersEquity",
                    "Liabilities",
                    "CommonStockSharesOutstanding",
                ]
                facts.append(
                    {
                        "cik": i + 1,
                        "namespace": "us-gaap",
                        "tag": tag,
                        "unit": (
                            "shares" if tag == "CommonStockSharesOutstanding" else "USD"
                        ),
                        "value": float(value),
                        "start": None if instant else f"{year}-01-01",
                        "end": f"{year}-12-31",
                        "date": f"{year}-12-31",
                        "available_at": pd.Timestamp(f"{year+1}-02-15T15:00:00Z"),
                        "accn": str(year),
                    }
                )
    entries = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    for day in entries.loc[entries >= pd.Timestamp("2010-01-01")]:
        for i in range(14):
            ticker = f"S{i:02d}" if i < 12 else ("NO_PRICE" if i == 12 else "NO_CIK")
            cohorts.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "cik": i + 1 if i < 13 else None,
                    "forecast_at": calendar.session_close(
                        calendar.previous_session(day)
                    )
                    + pd.Timedelta(hours=1),
                    "available_at": pd.Timestamp("2007-01-01T00:00:00Z"),
                }
            )
    for day in dates:
        for series, value in [
            ("DGS3MO", 1.0),
            ("DGS10", 4.0),
            ("DBAA", 6.0),
            ("VIXCLS", 20.0),
        ]:
            fred.append(
                {
                    "series_id": series,
                    "date": day,
                    "available_at": pd.Timestamp("2012-01-01T05:00:00Z"),
                    "value": value,
                }
            )
    fact_frame = pd.DataFrame(facts)
    fact_frame["start"] = pd.to_datetime(fact_frame["start"])
    fact_frame["end"] = pd.to_datetime(fact_frame["end"])
    store = ParquetStore(root / "data/processed")
    for name, frame in [
        ("prices", pd.DataFrame(prices)),
        ("company_facts", fact_frame),
        ("universe", pd.DataFrame(cohorts)),
        ("fred", pd.DataFrame(fred)),
        ("factors", pd.DataFrame(prices[:1])),
    ]:
        store.write(frame, name, "fixture")
    store.database(root / "data/ingen_qr.duckdb")
    config = root / "week2_fixture.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "database": "data/ingen_qr.duckdb",
                "output_directory": "data/week2",
                "development_start": "2010-01-01",
                "development_end": "2011-06-30",
                "first_test_month": "2011-03-01",
                "last_test_month": "2011-05-01",
                "market_proxy": "SPY",
                "financial_max_age_days": 550,
                "minimum_train_rows": 5,
            }
        ),
        encoding="utf-8",
    )
    (root / "docs").mkdir()
    (root / "docs/W02_Feature_Dictionary.md").write_text(
        "# Dictionary\n", encoding="utf-8"
    )
    return config


def test_complete_week2_from_real_duckdb_views_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_fixture(tmp_path)
    source_paths = list((tmp_path / "data/processed/fred").rglob("*.parquet"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    monkeypatch.chdir(tmp_path)
    result = run(config)
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert import_module("mlflow.telemetry").get_telemetry_client() is None
    assert result["feature_count"] == len(FEATURES) == 43
    assert result["firm_feature_count"] == len(FIRM_FEATURES) == 38
    assert result["macro_condition_count"] == len(MACRO_FEATURES) == 6
    assert result["holdout_opened"] is False
    raw = pd.read_parquet("data/week2/features_raw.parquet")
    assert "NO_CIK" not in set(raw["ticker"])
    assert "NO_PRICE" in set(raw["ticker"])
    processed = pd.read_parquet("data/week2/features_processed.parquet")
    assert processed[FEATURES].notna().all().all()
    conditions = pd.read_parquet("data/week2/macro_conditions.parquet")
    assert not conditions["date"].duplicated().any()
    assert not set(MACRO_FEATURES).intersection(processed.columns)
    assert not set(MACRO_FEATURES).intersection(raw.columns)
    reference = processed.merge(
        conditions[["date", *MACRO_FEATURES]], on="date", validate="many_to_one"
    )
    for name, (firm, macro) in INTERACTIONS.items():
        np.testing.assert_allclose(reference[name], reference[firm] * reference[macro])
    quality = pd.read_csv("data/week2/reports/feature_quality.csv")
    assert quality["feature"].nunique() == 43
    assert not set(MACRO_FEATURES).intersection(quality["feature"])
    breaks = pd.read_csv("data/week2/reports/coverage_breaks.csv")
    assert "gross_margin" in set(breaks["feature"])
    report = Path("docs/W02_Data_Quality_Report.md").read_text()
    assert "## Feature coverage breaks" in report
    assert "## Macro condition coverage and staleness" in report
    for feature in breaks["feature"].unique():
        assert str(feature) in report.split("## Feature coverage breaks")[1]
    assert not any(
        "cik" in str(c).lower()
        for c in processed.columns
        if str(c).startswith("unreported_")
    )
    scored = pd.read_csv("data/week2/reports/benchmark_predictions.csv")
    assert not scored.duplicated(["model", "date", "ticker"]).any()
    assert set(scored["model"]) == {"ols3", "momentum"}
    assert pd.to_datetime(scored["label_end"]).max() < pd.Timestamp("2024-01-01")
    assert all(
        p.exists()
        for p in [
            Path("docs/W02_Benchmark_Report.md"),
            Path("docs/W02_Data_Quality_Report.md"),
            Path("docs/figures/W02_Feature_Coverage.png"),
            Path("mlruns/tracking.db"),
        ]
    )
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    assert (
        str(result["mlflow_run_id"]) in Path("docs/W02_Benchmark_Report.md").read_text()
    )
