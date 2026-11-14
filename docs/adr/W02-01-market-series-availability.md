# W02-01: First archived vintage and market-series availability

Status: implemented research assumption; supervisor confirmation pending.

Decision: use earliest non-missing archived DGS3MO, DGS10, DBAA and VIXCLS values. Set derived availability to the next XNYS open strictly after the observation date. Calculate term spread as DGS10 minus DGS3MO and credit spread as DBAA minus DGS10, using matched observation dates. Do not use T10Y3M or BAA10Y.

Evidence supplied by the researcher for 2010-2023: multiple values occurred on 0.25% of DBAA dates, 0.11% of VIXCLS dates, 0.08% of DGS10 dates and no DGS3MO dates. Large ALFRED archive lags for some series motivate a distinct declared market-data assumption. Low revision frequency does not prove exact historical publication times or make the earliest archive an independently verified original release.

Implementation is isolated in features/macro.py. fred.py, storage.py, the original fred view and its Parquet files are untouched. The derived market file retains alfred_available_at and an assumed-availability flag. Monthly source dates, staleness and revision counts are exported. No future revised value replaces an earlier archived value.

Consequence: current development results depend on the assumption and are provisional on supervisor confirmation. Week 7 must compare against original ALFRED availability under a frozen configuration; raw data is preserved for that test. This ADR does not claim the assumption has been confirmed.


## Conditional features and normalization

The 38 firm characteristics alone undergo per-date winsorization, median imputation and rank normalization. Five interactions then multiply normalized firm values by the raw macro conditions; the products are never ranked again. Six macro levels remain in a separate once-per-date conditions file and do not count toward the 43 model features. Within-date ranking of common macro levels would erase them, while ranking a scalar interaction would discard its magnitude; the chosen ordering preserves the intended cross-month conditioning. Products may remain correlated with their inputs within a given date and are not claimed to be independent signals. Missing conditions propagate to products and are reported. Raw ALFRED provenance and the supervisor-confirmation status are unchanged.
