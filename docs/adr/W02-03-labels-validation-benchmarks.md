# W02-03: Labels, validation, embargo and benchmarks

Status: implements the W01 design and records operational details.

Feature origin is the previous XNYS close. The existing forecast cutoff is that close plus one hour, before first-session monthly opening execution. Label return uses adjusted_open on entry and adjusted_close on the 21st session, entry included. Risk-free accrual uses the forecast-known DGS3MO annual decimal quote and actual elapsed calendar time / 365, with simple accrual. Missing terminal prices, including delistings, remain missing. Classification retains top/bottom thirds and leaves the middle third unavailable.

Development is 2010-2023; 2015 is calibration; monthly OOS begins in January 2016. A label ending in 2024 is not a development observation. Split metadata contains expanding training, 12 months of validation and one test month, stepping monthly. Interval purging and a full pre-block calendar month after label completion are enforced. This conservative pre-block embargo is explicit: forward-only training has no post-validation training observations to which a post-block embargo could apply. The untouched holdout remains locked until an explicit later unlock.

OLS-3 is untuned and refits on all mature pre-test observations passing that embargo, including earlier validation months. The pure momentum benchmark uses the same scored firms/months. No OOS observation is duplicated. Both monthly signal series, ICIR, six-lag Newey-West statistics, OLS directional hit rate, and gross decile spreads are logged in the first MLflow experiment run. No ML model or portfolio backtest is run. Current eligibility and source-coverage limitations remain visible; benchmark results are not claims about an unbiased full S&P 500 universe.
