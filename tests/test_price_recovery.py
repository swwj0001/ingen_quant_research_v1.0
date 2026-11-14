from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingen_qr.data import price_audit


def test_cached_recovery_ignores_invalid_future_prices_but_restores_future_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports, raw = tmp_path / "reports", tmp_path / "raw"
    reports.mkdir()
    raw.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "provider": "TEST",
                "status": "Missing or invalid prices for TEST.",
            }
        ]
    ).to_csv(reports / "price_coverage.csv", index=False)
    dates = pd.DatetimeIndex(
        ["2023-01-30", "2023-01-31", "2024-01-03"], tz="America/New_York"
    )
    source = pd.DataFrame(
        {name: [50.0, 51.0, np.nan] for name in ["Open", "High", "Low", "Close"]},
        index=dates,
    )
    source["Volume"], source["Dividends"], source["Stock Splits"] = (
        2000.0,
        0.0,
        [0.0, 0.0, 2.0],
    )
    source.attrs = {"retrieved_at": "2024-01-04T23:00:00Z"}
    source.to_parquet(raw / "TEST_2008-01-01_2024-01-05.parquet")
    monkeypatch.setattr(
        price_audit,
        "load_price_source",
        lambda *a, **k: pytest.fail("Unexpected network"),
    )
    empty = pd.DataFrame(columns=["date", "ticker", "available_at"])
    result, audit = price_audit.recover_development_prices(
        empty, reports, raw, "2023-12-31"
    )
    assert result["raw_close"].tolist() == [100.0, 102.0]
    assert result["volume"].tolist() == [1000.0, 1000.0]
    assert audit["result"].item() == "recovered_development_rows"


def test_network_retry_is_opt_in_bounded_and_does_not_apply_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {"ticker": "OLD", "provider": "OLD", "status": "YFTzMissingError"},
            {"ticker": "LIVE", "provider": "LIVE", "status": "YFPricesMissingError"},
        ]
    ).to_csv(reports / "price_coverage.csv", index=False)
    calls = []

    def fake(ticker, start, end, downloader):
        calls.append((ticker, start, end))
        raise ValueError("Still missing")

    monkeypatch.setattr(price_audit, "load_price_source", fake)
    empty = pd.DataFrame(columns=["date", "ticker", "available_at"])
    price_audit.recover_development_prices(
        empty, reports, tmp_path / "raw", "2023-12-31"
    )
    assert not calls
    _, audit = price_audit.recover_development_prices(
        empty, reports, tmp_path / "raw", "2023-12-31", allow_network=True
    )
    assert calls == [("LIVE", "2008-01-01", "2024-01-01")]
    assert len(audit) == 2
