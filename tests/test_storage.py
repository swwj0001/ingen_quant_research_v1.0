from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ingen_qr.data.storage import ParquetStore, as_of


def test_datetime_precision_and_unchanged_input(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.to_datetime(["2014-12-31"]).as_unit("ns"),
            "available_at": pd.to_datetime(["2014-12-31T21:30:00Z"], utc=True).as_unit(
                "ns"
            ),
            "filed": pd.to_datetime(["2014-12-30T12:00:00.123456789"]).as_unit("ns"),
            "start": pd.to_datetime([None]).as_unit("ns"),
            "adjusted_close": [100.0],
        }
    )
    original = data.copy(deep=True)
    store = ParquetStore(tmp_path)
    store.write(data, "prices", "sample")
    parquet = next((tmp_path / "prices").rglob("*.parquet"))
    schema = pq.read_schema(parquet)
    for column in ("date", "available_at", "filed", "start"):
        assert schema.field(column).type.unit == "us"
    assert schema.field("available_at").type.tz == "UTC"
    stored = pd.read_parquet(parquet)
    assert stored["filed"].item() == pd.Timestamp("2014-12-30T12:00:00.123456")
    assert pd.isna(stored["start"].item())
    pd.testing.assert_frame_equal(data, original)


def test_parquet_asof_respects_revision_availability(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2014-12-29", "2014-12-29"]).as_unit("ns"),
            "available_at": pd.to_datetime(
                ["2014-12-29T21:30:00Z", "2014-12-30T21:30:00Z"], utc=True
            ).as_unit("ns"),
            "adjusted_close": [100.0, 999.0],
        }
    )
    store = ParquetStore(tmp_path)
    store.write(data, "prices", "sample")
    parquet = next((tmp_path / "prices").rglob("*.parquet"))
    with duckdb.connect() as connection:
        connection.read_parquet(str(parquet)).create_view("prices")
        assert (
            connection.execute(
                "SELECT typeof(available_at) FROM prices LIMIT 1"
            ).fetchone()[0]
            == "TIMESTAMP WITH TIME ZONE"
        )
        assert as_of(connection, "prices", "2014-12-29T21:29:59.999999Z").empty
        first = as_of(connection, "prices", "2014-12-29T21:30:00Z")
        assert first["adjusted_close"].item() == 100.0
        later = as_of(connection, "prices", "2014-12-30T21:30:00Z")
        assert later["adjusted_close"].item() == 999.0


@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Asia/Shanghai"])
def test_asof_date_comparison_uses_utc(tmp_path: Path, zone: str) -> None:
    data = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.to_datetime(["2014-12-30"]),
            "available_at": pd.to_datetime(["2014-12-30T00:01:00Z"], utc=True),
            "adjusted_close": [100.0],
        }
    )
    store = ParquetStore(tmp_path)
    store.write(data, "prices", "sample")
    parquet = next((tmp_path / "prices").rglob("*.parquet"))
    with duckdb.connect() as connection:
        connection.read_parquet(str(parquet)).create_view("prices")
        connection.execute(f"SET TimeZone='{zone}'")
        result = as_of(connection, "prices", "2014-12-30T00:05:00Z")
        assert result["adjusted_close"].item() == 100.0
        assert (
            connection.execute("SELECT current_setting('TimeZone')").fetchone()[0]
            == "UTC"
        )
