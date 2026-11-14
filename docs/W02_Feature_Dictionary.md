# W02 Feature Dictionary

Research only. Batch 1 contains four features; the complete Week 2 library is
still in progress. Coverage on the research universe is pending the full build.

## Common rules

- Source: the W01 `prices` table, using `adjusted_close`, a forward total-return
  index with an initial value of 100. Ratios include the W01 dividend/split logic.
- `P(t-k)` means the total-return index k XNYS trading sessions before t. A month
  is approximated by 21 sessions; three, six and twelve months by 63, 126 and 252.
- Observed rows are reindexed to the XNYS session grid before shifting. Missing
  dates never shorten a horizon and are never forward-filled.
- Return calculations require both price endpoints. Missing interior quotes do
  not prevent an endpoint return if both total-return index endpoints exist.
- A missing endpoint or insufficient history produces NaN, not zero. Missing
  feature handling is deferred to the cross-sectional preprocessing module.
- Features are assigned to t and available at the source row's `available_at`
  (W01 assumes XNYS close plus 30 minutes). Each required endpoint must have
  `available_at` no later than that timestamp. Execution must occur afterward.
- Supply one as-of price record per ticker/date. The function rejects duplicate
  snapshots. It does not reconstruct older revisions from a future data download;
  the W01 source/revision limitations still apply.
- Universe membership is applied in the subsequent full-panel builder. AAPL in
  the smoke check is a software check, not the research universe or a benchmark.

| Feature | Formula | Required lag endpoints | Direction | Coverage |
| --- | --- | --- | --- | --- |
| reversal_21d | -(P(t) / P(t-21) - 1) | 0 and 21 sessions | Larger means a larger recent decline | Pending full-panel build |
| momentum_63_21d | P(t-21) / P(t-63) - 1 | 21 and 63 sessions | Larger means stronger trailing momentum | Pending full-panel build |
| momentum_126_21d | P(t-21) / P(t-126) - 1 | 21 and 126 sessions | Larger means stronger trailing momentum | Pending full-panel build |
| momentum_252_21d | P(t-21) / P(t-252) - 1 | 21 and 252 sessions | Larger means stronger trailing momentum | Pending full-panel build |

## Checks

Hand-computed endpoint ratios; warm-up missingness; independent tickers;
unchanged input; missing-session alignment; future-price invariance; exclusion
of unavailable historical endpoints; duplicate-snapshot rejection; read-only
DuckDB integration restricted to data available through 2014-12-31.
The database check compares availability in UTC epoch nanoseconds, supporting
both microsecond and nanosecond time-zone-aware columns without truncating the
availability boundary.
