from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest
import requests
import yaml
from pandas.testing import assert_frame_equal

from ingen_qr import pipeline
from ingen_qr.data.download import Downloader
from ingen_qr.data.factors import _parse_daily_zip
from ingen_qr.data.fred import normalize_fred
from ingen_qr.data.membership import audit_membership, build_cohorts
from ingen_qr.data.price_panel import _prepare_price_panel
from ingen_qr.data.sec import normalize_company_facts
from ingen_qr.data.storage import ParquetStore, as_of


def price_source(
    days: list[str], closes: list[float], splits: list[float], captured: str
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {name: closes for name in ("Open", "High", "Low", "Close")},
        index=pd.DatetimeIndex(days, tz="America/New_York"),
    )
    frame["Volume"] = 1200.0
    frame["Dividends"] = 0.0
    frame["Stock Splits"] = splits
    frame.attrs = {"retrieved_at": captured}
    return frame


def test_retry_then_disk_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader(str(tmp_path), min_interval=0)
    calls = []

    def fetch() -> bytes:
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("temporary")
        return b"downloaded"

    waits = []
    monkeypatch.setattr("ingen_qr.data.download.time.sleep", waits.append)
    monkeypatch.setattr("ingen_qr.data.download.random.uniform", lambda *_: 0.0)
    assert downloader.cached("one", fetch) == b"downloaded"
    assert waits == [1.0]
    other = Downloader(str(tmp_path), min_interval=0)
    assert (
        other.cached("one", lambda: pytest.fail("Network on cache hit"))
        == b"downloaded"
    )
    assert len(calls) == 2


def test_future_split_does_not_change_past_prices() -> None:
    original = price_source(
        ["2010-01-04", "2010-01-05"],
        [100.0, 102.0],
        [0.0, 0.0],
        "2010-01-05T23:00:00Z",
    )
    later = price_source(
        ["2010-01-04", "2010-01-05", "2010-01-06"],
        [50.0, 51.0, 52.0],
        [0.0, 0.0, 2.0],
        "2010-01-06T23:00:00Z",
    )
    later.loc[later.index[:2], "Volume"] *= 2.0
    first = _prepare_price_panel(original, "TEST")
    second = _prepare_price_panel(later, "TEST").iloc[:2].reset_index(drop=True)
    assert_frame_equal(first, second, check_exact=False, rtol=1e-12)


def test_early_close_and_incomplete_session() -> None:
    source = price_source(
        ["2010-11-24", "2010-11-26"],
        [100.0, 101.0],
        [0.0, 0.0],
        "2010-11-26T18:15:00Z",
    )
    assert len(_prepare_price_panel(source, "TEST")) == 1
    source.attrs["retrieved_at"] = "2010-11-26T19:00:00Z"
    result = _prepare_price_panel(source, "TEST")
    assert result["available_at"].iloc[-1] == pd.Timestamp("2010-11-26T18:30:00Z")


def test_factor_percent_and_missing_codes() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "factor.csv", "Preamble\n,Mom\n20100104,1.25\n20100105,-99.99\n2010,99\n"
        )
    result = _parse_daily_zip(buffer.getvalue())
    assert len(result) == 2
    assert result["mom"].iloc[0] == 0.0125
    assert pd.isna(result["mom"].iloc[1])


def test_no_look_ahead_with_fred_revisions(tmp_path: Path) -> None:
    observations = [
        {
            "date": "2010-01-04",
            "value": "1.0",
            "realtime_start": "2010-01-05",
            "realtime_end": "2010-02-01",
        },
        {
            "date": "2010-01-04",
            "value": "9.0",
            "realtime_start": "2010-02-02",
            "realtime_end": "9999-12-31",
        },
    ]
    frame = normalize_fred(observations, "TEST")
    store = ParquetStore(tmp_path / "panel")
    store.write(frame, "fred", "test")
    with duckdb.connect() as connection:
        pattern = (tmp_path / "panel/fred/**/*.parquet").as_posix()
        connection.execute(
            f"CREATE VIEW fred AS SELECT * FROM read_parquet('{pattern}')"
        )
        assert as_of(connection, "fred", "2010-01-05T23:00:00Z").empty
        assert as_of(connection, "fred", "2010-01-07T12:00:00Z")["value"].item() == 1.0
        assert as_of(connection, "fred", "2010-02-04T12:00:00Z")["value"].item() == 9.0


def company_payload(cik: int = 320193) -> dict:
    return {
        "cik": cik,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "end": "2022-12-31",
                                "filed": "2023-01-06",
                                "accn": "1",
                                "form": "10-K",
                            },
                            {
                                "val": 120,
                                "end": "2022-12-31",
                                "filed": "2023-02-03",
                                "accn": "2",
                                "form": "10-K/A",
                            },
                        ]
                    }
                }
            }
        },
    }


def test_sec_filing_lag_and_restatement_are_preserved() -> None:
    result = normalize_company_facts(company_payload(), ["Assets"])
    assert len(result) == 2
    assert result["available_at"].iloc[0] == pd.Timestamp("2023-01-09T14:30:00Z")
    assert result["available_at"].iloc[1] == pd.Timestamp("2023-02-06T14:30:00Z")
    assert result["value"].tolist() == [100.0, 120.0]
    assert result["is_amendment"].tolist() == [False, True]


def membership_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-12-18", "2020-12-21", "2023-01-03", "2023-01-04"]
            ),
            "tickers": [
                "AAPL,IBM,MSFT",
                "AAPL,IBM,MSFT,TSLA",
                "AAPL,IBM,MSFT,TSLA",
                "AAPL,IBM,MSFT,TSLA,GEHC",
            ],
        }
    )


def test_membership_uses_information_before_forecast() -> None:
    source = membership_source()
    assert audit_membership(source)["matches"].all()
    cohorts = build_cohorts(source, "2023-01-01", "2023-01-31", "2026-09-07")
    assert "GEHC" not in set(cohorts["ticker"])
    assert (cohorts["available_at"] <= cohorts["forecast_at"]).all()


def test_storage_rejects_missing_available_at(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    data = pd.DataFrame(
        {"date": [pd.Timestamp("2010-01-04")], "available_at": [pd.NaT]}
    )
    with pytest.raises(ValueError, match="Missing available_at"):
        store.write(data, "prices", "test")


def test_complete_build_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pipeline.yaml").read_text("utf-8"))
    config.update(study_start="2023-01-01", study_end="2023-01-31")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "test-key-not-used-on-network")
    monkeypatch.setenv("SEC_USER_AGENT", "Test researcher@test.invalid")
    monkeypatch.setattr(
        pipeline, "load_constituent_source", lambda **_: membership_source()
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-30", "2023-01-03"]),
            "available_at": pd.to_datetime(
                ["2022-12-30T21:30:00Z", "2023-01-03T21:30:00Z"]
            ),
            "adjusted_close": [100.0, 101.0],
        }
    )
    monkeypatch.setattr(pipeline, "load_prices", lambda *_, **__: prices.copy())
    monkeypatch.setattr(pipeline, "load_factors", lambda *_, **__: prices.copy())
    fred = [
        {
            "date": "2023-01-03",
            "value": "4.0",
            "realtime_start": "2023-01-04",
            "realtime_end": "9999-12-31",
        }
    ]
    monkeypatch.setattr(
        pipeline, "load_fred", lambda series, *_: normalize_fred(fred, series)
    )
    monkeypatch.setattr(
        pipeline,
        "load_ticker_map",
        lambda *_, **__: {"AAPL": 320193, "MSFT": 789019, "IBM": 51143},
    )
    monkeypatch.setattr(
        pipeline,
        "load_company_facts",
        lambda cik, tags, *_: normalize_company_facts(company_payload(cik), tags),
    )
    pipeline.build(config, smoke=True, refresh=False)
    assert (tmp_path / "data/reports/build_report.json").exists()
    with duckdb.connect(
        str(tmp_path / "data/ingen_qr.duckdb"), read_only=True
    ) as connection:
        names = set(connection.execute("SHOW TABLES").df()["name"])
        assert names == {"prices", "fred", "factors", "company_facts", "universe"}
        assert connection.execute("SELECT count(*) FROM prices").fetchone()[0] == 6
