import re
import shutil
from pathlib import Path

import duckdb
import pandas as pd

TABLES = {"prices", "factors", "fred", "company_facts", "universe"}
KEYS = {
    "prices": ["ticker", "date"],
    "fred": ["series_id", "date"],
    "factors": ["date"],
    "company_facts": ["cik", "namespace", "tag", "unit", "start", "end"],
}


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.written: set[str] = set()

    def reset(self) -> None:
        for table in TABLES:
            directory = self.root / table
            if directory.exists():
                shutil.rmtree(directory)

    def write(self, frame: pd.DataFrame, table: str, part: str) -> None:
        if table not in TABLES or not re.fullmatch(r"[A-Za-z0-9_.-]+", part):
            raise ValueError("Invalid table or part name.")
        if frame.empty:
            return
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        data["available_at"] = pd.to_datetime(data["available_at"], utc=True)
        if data["available_at"].isna().any():
            raise ValueError(f"Missing available_at in {table}.")
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = data[column].dt.as_unit("us")
        partitions = data["date"].dt.strftime("%Y-%m-01")
        for date, group in data.groupby(partitions):
            directory = self.root / table / f"partition_date={date}"
            directory.mkdir(parents=True, exist_ok=True)
            group.to_parquet(directory / f"part-{part}.parquet", index=False)
        self.written.add(table)

    def database(self, path: Path) -> None:
        missing = TABLES - self.written
        if missing:
            raise ValueError(f"Required tables have no records: {sorted(missing)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(path)) as connection:
            for table in sorted(TABLES):
                pattern = (self.root / table).resolve().as_posix() + "/**/*.parquet"
                pattern = pattern.replace("'", "''")
                connection.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM "
                    f"read_parquet('{pattern}', hive_partitioning=true, "
                    "union_by_name=true)"
                )


def as_of(
    connection: duckdb.DuckDBPyConnection, table: str, cutoff: str
) -> pd.DataFrame:
    if table not in KEYS:
        raise ValueError("Use an explicit cohort date to query the universe table.")
    connection.execute("SET TimeZone='UTC'")
    keys = ", ".join(f'"{key}"' for key in KEYS[table])
    extra_order = ", accn DESC" if table == "company_facts" else ""
    return connection.execute(
        f"SELECT * FROM {table} WHERE available_at <= ? "
        "AND date <= CAST(? AS TIMESTAMPTZ) "
        f"QUALIFY row_number() OVER (PARTITION BY {keys} "
        f"ORDER BY available_at DESC{extra_order}) = 1",
        [pd.Timestamp(cutoff).to_pydatetime()] * 2,
    ).df()
