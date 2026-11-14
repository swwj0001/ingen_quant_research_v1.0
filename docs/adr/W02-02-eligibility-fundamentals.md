# W02-02: Eligibility and financial reconstruction

Status: implemented under the researcher's fixed rules.

Exclude companies with no CIK uniformly across all historical cohorts and report the excluded count/share by month. Preserve the original universe in the source database. This uses the supplied current-reference mapping and may still exclude firms because of later delisting or corporate changes; uniform application does not eliminate survival-related bias. Missing-CIK indicators are not used.

Use only us-gaap/dei facts, queried through storage.as_of at each forecast cutoff. Reconstruct duration periods from start/end and the filing vintage, never frame. Annual totals or four adjacent discrete quarters form TTM flows; YTD differences produce Q2/Q3, and annual minus nine-month YTD produces Q4. Stock values use their latest report. Retain the existing next-session-after-filed-date availability proxy and report its limitation relative to exact acceptance timestamps.

Prefer dei shares, then us-gaap shares, adjusting observed splits through the feature origin. Reported shares are not verified free float. Concurrent share classes with one CIK have unavailable market-cap/value proxies. A 550-calendar-day period-end age limit and a 30-day fiscal matching tolerance are explicit engineering choices. Division by non-positive denominators produces missing values.

Proposed symbol renames are listed separately with source links and a pending-review status. They are not automatically applied. Acquisitions, bankruptcies and privatizations must not be mapped to a successor's stock return series.
