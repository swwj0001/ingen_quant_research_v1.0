# W02 Feature Dictionary

Research only. Batches 1-2 contain 11 features. Full-universe coverage remains
pending the complete feature-panel build.

## Common rules

- `P(t)` is W01 `prices.adjusted_close`, a forward total-return index including
  the W01 split/dividend treatment. `r(t) = P(t)/P(t-1)-1` requires prices on two
  adjacent XNYS trading sessions; it never bridges a missing session or fills a
  price forward. Calendar holidays are not missing trading sessions.
- A month is approximated by 21 sessions; three, six and twelve months by 63,
  126 and 252. Momentum uses endpoint ratios. Risk and liquidity windows contain
  the last N XNYS sessions ending at t. A daily return at the window's first
  session also needs the immediately preceding session's price.
- Batch 2 requires at least ceil(0.8*N) valid daily observations: 17/21, 48/60,
  202/252. This also applies during initial warm-up; there is no separate
  requirement to wait for N observed rows. Batch 1 retains its endpoint rules.
- Beta and idiosyncratic volatility require the same 202 or more paired stock/SPY
  daily returns. SPY total returns are the market proxy, matching the large-cap
  US equity universe. They are raw total returns, not excess returns over cash.
- Fama-French snapshots are deliberately excluded from all features. W01 sets
  their `available_at` to retrieval time and marks them `ex_post_attribution_only`;
  the current revised snapshot is not a historical real-time observation.
- An output row has the observed stock row's exact UTC `available_at`. Every
  input to that row must have `available_at <=` that cutoff, including both
  endpoints of each stock or SPY return. Late historical observations are
  excluded separately at each output cutoff, retaining timestamp precision.
- Output is available after W01's assumed XNYS close plus 30 minutes. Trading
  execution must occur afterward. Source revision limitations still apply.
- Standard deviations use ddof=1. Idiosyncratic volatility is the sample standard
  deviation of OLS residuals with an intercept, sqrt(SSE/(n-1)), rather than the
  regression standard error sqrt(SSE/(n-2)). All returns/volatilities are daily
  decimal values; none are annualized.
- Zero-volume days are excluded from Amihud but count as zero dollar volume for
  the mean dollar-volume feature. A zero mean dollar volume produces NaN for its
  log. A zero market-return variance produces NaN for beta and idiosyncratic
  volatility. Missing values are handled later by cross-sectional preprocessing.
- The pipeline loads SPY in both full and smoke mode. It remains outside the
  stock universe, CIK requests and stock coverage counts. The feature functions
  may emit SPY diagnostic rows; the later universe join determines model rows.

| Feature | Formula | Source | Lag/window rule | Unit | Direction | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| reversal_21d | -(P(t)/P(t-21)-1) | prices.adjusted_close | Endpoints 0,21; both required | Decimal return | Larger means a larger recent decline | Pending full-panel build |
| momentum_63_21d | P(t-21)/P(t-63)-1 | prices.adjusted_close | Endpoints 21,63; both required | Decimal return | Larger means stronger trailing momentum | Pending full-panel build |
| momentum_126_21d | P(t-21)/P(t-126)-1 | prices.adjusted_close | Endpoints 21,126; both required | Decimal return | Larger means stronger trailing momentum | Pending full-panel build |
| momentum_252_21d | P(t-21)/P(t-252)-1 | prices.adjusted_close | Endpoints 21,252; both required | Decimal return | Larger means stronger trailing momentum | Pending full-panel build |
| volatility_21d | sd(r), ddof=1 | prices.adjusted_close | Last 21 sessions; >=17 returns | Daily decimal return | Larger means higher daily variability | Pending full-panel build |
| volatility_60d | sd(r), ddof=1 | prices.adjusted_close | Last 60 sessions; >=48 returns | Daily decimal return | Larger means higher daily variability | Pending full-panel build |
| beta_252d | cov(r,SPY r)/var(SPY r), intercept included | Stock and SPY prices.adjusted_close | Last 252 sessions; >=202 pairs | Dimensionless | Market sensitivity | Pending full-panel build |
| idio_volatility_252d | sd(r-alpha-beta*SPY r), ddof=1 | Stock and SPY prices.adjusted_close | Same 252-session regression and paired sample | Daily decimal return | Larger means higher residual variability | Pending full-panel build |
| max_return_21d | max(r) | prices.adjusted_close | Last 21 sessions; >=17 returns | Daily decimal return | Largest observed daily gain | Pending full-panel build |
| amihud_21d | mean(abs(r)/(raw_close*volume))*1e6 | prices.adjusted_close, raw_close, volume | Last 21 sessions; >=17 valid positive-volume observations | Absolute decimal return per USD million traded | Larger means less liquid | Pending full-panel build |
| log_dollar_volume_21d | ln(mean(raw_close*volume)/(1 USD)) | prices.raw_close, volume | Last 21 sessions; >=17 dollar-volume observations | Log USD; dimensionless after USD normalization | Larger means more dollar trading | Pending full-panel build |

Coverage is computed on the actual universe in the later data-quality report;
training-only AAPL/SPY checks do not establish coverage of the full stock pool.
Turnover is deferred until point-in-time shares outstanding are available.
