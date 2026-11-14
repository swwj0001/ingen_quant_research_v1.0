"""Read-only training-period check of both batches for AAPL and SPY."""

from pathlib import Path

import duckdb

from ingen_qr.features.liquidity import FEATURES as LIQUIDITY
from ingen_qr.features.liquidity import liquidity_features
from ingen_qr.features.momentum import FEATURES as MOMENTUM
from ingen_qr.features.momentum import momentum_features
from ingen_qr.features.risk import FEATURES as RISK
from ingen_qr.features.risk import risk_features


def main() -> None:
    cutoff = "2014-12-31T23:59:59Z"
    with duckdb.connect(
        str(Path("data/ingen_qr.duckdb")), read_only=True
    ) as connection:
        prices = connection.execute(
            "SELECT ticker, date, adjusted_close, raw_close, volume, available_at "
            "FROM prices WHERE ticker IN ('AAPL', 'SPY') "
            "AND date <= CAST(? AS DATE) "
            "AND epoch_ns(available_at) <= epoch_ns(CAST(? AS TIMESTAMPTZ)) "
            "QUALIFY row_number() OVER (PARTITION BY ticker, date "
            "ORDER BY available_at DESC) = 1 ORDER BY ticker, date",
            [cutoff, cutoff],
        ).df()
    missing = {"AAPL", "SPY"} - set(prices["ticker"])
    if missing:
        raise ValueError(f"Missing prices for {sorted(missing)}; rebuild the pipeline.")
    result = momentum_features(prices).merge(
        risk_features(prices),
        on=["ticker", "date", "available_at"],
        validate="one_to_one",
    )
    result = result.merge(
        liquidity_features(prices),
        on=["ticker", "date", "available_at"],
        validate="one_to_one",
    )
    features = [*MOMENTUM, *RISK, *LIQUIDITY]
    print("Research only: AAPL/SPY training-period feature check.")
    print("Cutoff:", cutoff)
    for ticker in ("AAPL", "SPY"):
        selected = result.loc[result["ticker"] == ticker]
        print("Ticker:", ticker, "Rows:", len(selected))
        print(selected.tail(3).to_string(index=False))
        print(selected[features].notna().sum().to_string())


if __name__ == "__main__":
    main()
