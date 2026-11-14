"""Daily risk features using a point-in-time SPY total-return proxy."""

import numpy as np
import pandas as pd

from ingen_qr.features.common import (
    daily_returns,
    minimum_count,
    output_rows,
    prepare_prices,
    price_blocks,
    timestamp_ns,
    window_std,
    window_values,
)

FEATURES = [
    "volatility_21d",
    "volatility_60d",
    "beta_252d",
    "idio_volatility_252d",
    "max_return_21d",
]


def risk_features(prices: pd.DataFrame, market_proxy: str = "SPY") -> pd.DataFrame:
    frame = prepare_prices(prices)
    market = frame.loc[frame["ticker"] == market_proxy].set_index("date")
    if market.empty:
        raise ValueError(
            f"Missing market proxy {market_proxy}; rebuild the prices table."
        )

    results: list[pd.DataFrame] = []
    for ticker, block, observed in price_blocks(frame):
        returns, known = daily_returns(block)
        cutoffs = timestamp_ns(block["available_at"])
        for width in (21, 60):
            values = window_values(returns, known, cutoffs, width)
            block[f"volatility_{width}d"] = window_std(values, width)
        short = window_values(returns, known, cutoffs, 21)
        count = np.isfinite(short).sum(axis=1)
        maximum = np.where(np.isfinite(short), short, -np.inf).max(axis=1)
        block["max_return_21d"] = np.where(count >= 17, maximum, np.nan)

        market_returns, market_known = daily_returns(market.reindex(block.index))
        stock = window_values(returns, known, cutoffs, 252)
        proxy = window_values(market_returns, market_known, cutoffs, 252)
        paired = np.isfinite(stock) & np.isfinite(proxy)
        count = paired.sum(axis=1)
        stock_mean = np.where(paired, stock, 0.0).sum(axis=1) / np.maximum(count, 1)
        proxy_mean = np.where(paired, proxy, 0.0).sum(axis=1) / np.maximum(count, 1)
        y = np.where(paired, stock - stock_mean[:, None], 0.0)
        x = np.where(paired, proxy - proxy_mean[:, None], 0.0)
        xx = (x * x).sum(axis=1)
        xy = (x * y).sum(axis=1)
        beta = np.divide(xy, xx, out=np.full(len(block), np.nan), where=xx > 0)
        valid = (count >= minimum_count(252)) & (xx > 0)
        block["beta_252d"] = np.where(valid, beta, np.nan)
        residuals = np.where(paired, y - beta[:, None] * x, 0.0)
        residual_std = np.sqrt(
            (residuals * residuals).sum(axis=1) / np.maximum(count - 1, 1)
        )
        block["idio_volatility_252d"] = np.where(valid, residual_std, np.nan)
        results.append(output_rows(block, observed, ticker, FEATURES))

    return pd.concat(results, ignore_index=True)
