"""Daily dollar-volume and Amihud liquidity features."""

import numpy as np
import pandas as pd

from ingen_qr.features.common import (
    daily_returns,
    output_rows,
    prepare_prices,
    price_blocks,
    timestamp_ns,
    window_mean,
    window_values,
)

FEATURES = ["amihud_21d", "log_dollar_volume_21d"]


def liquidity_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_prices(prices, ("raw_close", "volume"))
    results: list[pd.DataFrame] = []
    for ticker, block, observed in price_blocks(frame):
        returns, return_known = daily_returns(block)
        known = timestamp_ns(block["available_at"])
        dollars = (block["raw_close"] * block["volume"]).to_numpy(
            dtype=float, na_value=np.nan
        )
        amihud = np.divide(
            np.abs(returns) * 1e6,
            dollars,
            out=np.full(len(block), np.nan),
            where=dollars > 0,
        )
        values = window_values(amihud, return_known, known, 21)
        block["amihud_21d"] = window_mean(values, 21)
        dollar_window = window_values(dollars, known, known, 21)
        mean_dollars = window_mean(dollar_window, 21)
        logged = np.full(len(block), np.nan)
        np.log(mean_dollars, out=logged, where=mean_dollars > 0)
        block["log_dollar_volume_21d"] = logged
        results.append(output_rows(block, observed, ticker, FEATURES))
    return pd.concat(results, ignore_index=True)
