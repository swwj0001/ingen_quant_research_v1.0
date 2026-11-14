import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from ingen_qr import pipeline


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("fail_proxy", [False, True])
def test_proxy_is_loaded_but_not_counted_as_a_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke: bool, fail_proxy: bool
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pipeline.yaml").read_text("utf-8"))
    config.update(study_start="2023-01-01", study_end="2023-01-31")
    cohorts = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "IBM", "TSLA", "SPY"],
            "date": pd.Timestamp("2023-01-03"),
            "forecast_at": pd.Timestamp("2022-12-30T22:00:00Z"),
            "available_at": pd.Timestamp("2022-12-30T22:00:00Z"),
        }
    )
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-30", "2023-01-03"]),
            "available_at": pd.to_datetime(
                ["2022-12-30T21:30:00Z", "2023-01-03T21:30:00Z"]
            ),
            "adjusted_close": [100.0, 101.0],
            "raw_close": [100.0, 101.0],
            "volume": [1000.0, 1000.0],
        }
    )
    calls: list[str] = []

    def fake_prices(ticker: str, *_: object, **__: object) -> pd.DataFrame:
        calls.append(ticker)
        if fail_proxy and ticker == "SPY":
            raise ValueError("Missing proxy fixture")
        return data.copy()

    def deny_network(*_: object, **__: object) -> None:
        pytest.fail("Offline pipeline test attempted a network request.")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("requests.sessions.Session.request", deny_network)
    monkeypatch.setattr(
        pipeline, "credentials", lambda: ("unused", "test@test.invalid")
    )
    monkeypatch.setattr(pipeline, "load_constituent_source", lambda **_: pd.DataFrame())
    monkeypatch.setattr(pipeline, "audit_membership", lambda _: pd.DataFrame())
    monkeypatch.setattr(pipeline, "build_cohorts", lambda *_: cohorts.copy())
    monkeypatch.setattr(pipeline, "load_prices", fake_prices)
    monkeypatch.setattr(pipeline, "load_factors", lambda *_, **__: data.copy())
    monkeypatch.setattr(pipeline, "load_fred", lambda *_, **__: data.copy())
    monkeypatch.setattr(pipeline, "load_ticker_map", lambda *_, **__: {"TSLA": 1318605})
    monkeypatch.setattr(pipeline, "load_company_facts", lambda *_, **__: data.copy())
    if fail_proxy:
        with pytest.raises(RuntimeError, match="proxy SPY"):
            pipeline.build(config, smoke=smoke, refresh=False)
        return
    pipeline.build(config, smoke=smoke, refresh=False)

    count = 3 if smoke else 4
    assert calls.count("SPY") == 1
    assert set(calls) == set(cohorts["ticker"]) - ({"TSLA"} if smoke else set())
    summary = json.loads(Path("data/reports/build_report.json").read_text("utf-8"))
    assert summary["requested_stocks"] == count
    assert summary["stocks_with_prices"] == count
    assert summary["market_proxy"] == "SPY"
    assert summary["market_proxy_rows"] == 2
    coverage = pd.read_csv("data/reports/monthly_coverage.csv")
    assert coverage["pool_size"].item() == count
    assert coverage["feature_price_covered"].item() == count
    assert "SPY" not in set(pd.read_csv("data/reports/price_coverage.csv")["ticker"])
    with duckdb.connect("data/ingen_qr.duckdb", read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM prices WHERE ticker = 'SPY'"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM prices WHERE ticker <> 'SPY'"
            ).fetchone()[0]
            == 2 * count
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM universe WHERE ticker = 'SPY'"
            ).fetchone()[0]
            == 0
        )
