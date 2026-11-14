import os
import subprocess
import sys
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import numpy as np
import pandas as pd


def test_feature_script_reads_both_tickers_and_keeps_cutoff(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "data").mkdir()
    calendar = xcals.get_calendar("XNYS", start="2013-01-01", end="2014-12-31")
    dates = pd.DatetimeIndex(calendar.sessions).tz_localize(None)
    blocks = []
    for ticker, growth in [("AAPL", 0.002), ("SPY", 0.001)]:
        blocks.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "available_at": pd.to_datetime(dates, utc=True)
                    + pd.Timedelta(hours=23),
                    "adjusted_close": 100.0 * (1.0 + growth) ** np.arange(len(dates)),
                    "raw_close": 100.0,
                    "volume": 2000.0,
                }
            )
        )
    panel = pd.concat(blocks, ignore_index=True)
    with duckdb.connect(str(tmp_path / "data/ingen_qr.duckdb")) as connection:
        connection.register("panel", panel)
        connection.execute("CREATE TABLE prices AS SELECT * FROM panel")
        connection.execute(
            "INSERT INTO prices VALUES ('AAPL', DATE '2014-12-31', "
            "CAST('2014-12-31T23:59:59.000000001Z' AS TIMESTAMPTZ_NS), "
            "999999.0, 100.0, 2000.0)"
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(project / "scripts/check_features.py")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Ticker: AAPL Rows: 504" in result.stdout
    assert "Ticker: SPY Rows: 504" in result.stdout
    assert "999999" not in result.stdout
    assert "idio_volatility_252d" in result.stdout
    assert "amihud_21d" in result.stdout
