"""Read-only smoke check on AAPL prices from the initial training period."""

from pathlib import Path

import duckdb

from ingen_qr.features.momentum import FEATURES, momentum_features


def main() -> None:
    path = Path("data/ingen_qr.duckdb")
    cutoff = "2014-12-31T23:59:59Z"
    with duckdb.connect(str(path), read_only=True) as connection:
        prices = connection.execute(
            "SELECT ticker, date, adjusted_close, available_at FROM prices "
            "WHERE ticker = ? AND date <= CAST(? AS DATE) "
            "AND epoch_ns(available_at) <= epoch_ns(CAST(? AS TIMESTAMPTZ)) "
            "QUALIFY row_number() OVER (PARTITION BY ticker, date "
            "ORDER BY available_at DESC) = 1 ORDER BY date",
            ["AAPL", cutoff, cutoff],
        ).df()

    features = momentum_features(prices)
    print("Research only: AAPL training-period smoke check.")
    print("Cutoff:", cutoff)
    print("Rows:", len(features))
    print(features.tail(3).to_string(index=False))
    print("Non-missing feature counts:")
    print(features[list(FEATURES)].notna().sum().to_string())


if __name__ == "__main__":
    main()
