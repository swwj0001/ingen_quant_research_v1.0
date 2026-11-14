# W02 Feature Dictionary

Research only. 43 features (38 firm characteristics plus five conditional interactions), and six separately stored macro conditions. Missing-report indicators are not counted as economic features. All calculations use historical XNYS sessions and the configured historical monthly cohorts.

## Common rules

The feature origin is the previous XNYS session. The existing cohort forecast_at is the origin close plus one hour; monthly entry is the following first-session open. A feature input must be available no later than forecast_at. The output available_at is that conservative forecast cutoff. No sealed-holdout returns or model results are read by the W2 runner.

Daily return is adjusted_close(t)/adjusted_close(previous XNYS session)-1. A missing session remains a gap. No forward filling or cross-gap daily return is used. The existing momentum features compare the specified endpoints. Trailing daily statistics require at least 80% valid observations: 17/21, 48/60, and 202/252. Standard deviations use ddof=1 and daily units. Beta uses paired stock/SPY total returns with an intercept; residual volatility is sqrt(SSE/(n-1)). SPY is excluded from the research stock pool. Fama-French data is ex-post attribution only and is never a Beta input.

Companies with no CIK in the supplied mapping are excluded at every date. The monthly exclusion proportion is reported. This coverage screen can still create survival-related selection; applying it uniformly does not prove absence of survivorship bias. Unreported indicators are created only for absent required financial concepts in an otherwise available filing snapshot, never for a missing CIK, missing price history, a zero denominator, or a structurally ambiguous share class.

## Model features: 38 firm characteristics and five conditional interactions

| Feature | Formula | Source | Availability/lag rule | Unit |
| --- | --- | --- | --- | --- |
| reversal_21d | -(P(t)/P(t-21)-1) | Yahoo forward total-return adjusted_close | price cutoff | decimal return |
| momentum_63_21d | P(t-21)/P(t-63)-1 | Yahoo forward total-return adjusted_close | price cutoff | decimal return |
| momentum_126_21d | P(t-21)/P(t-126)-1 | Yahoo forward total-return adjusted_close | price cutoff | decimal return |
| momentum_252_21d | P(t-21)/P(t-252)-1 | Yahoo forward total-return adjusted_close | price cutoff | decimal return |
| volatility_21d | sample std of the last 21 daily returns | Yahoo adjusted_close; SPY total return for regression | price cutoff | daily decimal return; not annualized |
| volatility_60d | sample std of the last 60 daily returns | Yahoo adjusted_close; SPY total return for regression | price cutoff | daily decimal return; not annualized |
| beta_252d | paired stock/SPY covariance divided by SPY variance, with intercept | Yahoo adjusted_close; SPY total return for regression | price cutoff | dimensionless beta |
| idio_volatility_252d | sqrt(SSE/(paired n-1)) from the 252-session stock/SPY regression | Yahoo adjusted_close; SPY total return for regression | price cutoff | daily decimal return; not annualized |
| max_return_21d | maximum daily return in the last 21 sessions | Yahoo adjusted_close; SPY total return for regression | price cutoff | daily decimal return; not annualized |
| amihud_21d | mean(abs(daily return)/(raw_close x volume)) x 1e6; exclude zero volume | Yahoo raw_close, volume and total returns | price cutoff | return per USD million |
| log_dollar_volume_21d | ln(mean(raw_close x volume)) over 21 sessions | Yahoo raw_close and volume | price cutoff | log USD/day |
| log_market_cap | ln(raw close x split-adjusted reported shares) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | log USD |
| book_to_market | positive latest equity / market capitalization | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| earnings_yield | TTM net income / market capitalization | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| sales_to_price | TTM revenue / market capitalization | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| cashflow_yield | TTM operating cash flow / market capitalization | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| roe | TTM net income / mean(current, year-ago equity) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| roa | TTM net income / mean(current, year-ago assets) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| gross_profitability | TTM gross profit / latest assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| gross_margin | TTM gross profit / TTM revenue | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| net_margin | TTM net income / TTM revenue | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| operating_cashflow_to_assets | TTM operating cash flow / latest assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| accruals | (TTM net income - TTM operating cash flow) / average assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| asset_growth | latest assets / year-ago assets - 1 | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | decimal change |
| equity_growth | latest equity / year-ago equity - 1 | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | decimal change |
| sales_growth | TTM revenue / year-ago TTM revenue - 1 | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | decimal change |
| earnings_growth | (TTM net income - year-ago TTM income) / abs(year-ago TTM income) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | decimal change |
| liabilities_to_assets | latest liabilities / latest assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| equity_to_assets | latest equity / latest assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| liabilities_to_equity | latest liabilities / positive latest equity | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| capex_to_assets | TTM property/plant/equipment purchases / latest assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| capex_to_sales | TTM property/plant/equipment purchases / TTM revenue | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| free_cashflow_yield | (TTM operating cash flow - TTM capital expenditure) / market capitalization | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| asset_turnover | TTM revenue / average assets | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | ratio |
| log_assets | ln(latest positive assets) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | log USD |
| log_book_equity | ln(latest positive stockholders equity) | SEC us-gaap/dei; raw_close for valuation | filing cutoff; latest period <= origin | log USD |
| turnover_21d | mean(split-normalized daily volume)/latest as-of reported common shares; 21 sessions | Yahoo volume/split_ratio; SEC dei/us-gaap shares | price and filing cutoff | daily shares-traded ratio; not free-float turnover |
| turnover_60d | mean(split-normalized daily volume)/latest as-of reported common shares; 60 sessions | Yahoo volume/split_ratio; SEC dei/us-gaap shares | price and filing cutoff | daily shares-traded ratio; not free-float turnover |
| momentum_x_term | normalized momentum_252_21d x raw term_spread | corresponding firm and macro features | maximum input cutoff | normalized firm rank x raw macro unit |
| beta_x_credit | normalized beta_252d x raw credit_spread | corresponding firm and macro features | maximum input cutoff | normalized firm rank x raw macro unit |
| volatility_x_vix | normalized volatility_60d x raw vix_level | corresponding firm and macro features | maximum input cutoff | normalized firm rank x raw macro unit |
| value_x_term | normalized book_to_market x raw term_spread | corresponding firm and macro features | maximum input cutoff | normalized firm rank x raw macro unit |
| size_x_vix | normalized log_market_cap x raw vix_level | corresponding firm and macro features | maximum input cutoff | normalized firm rank x raw macro unit |

## Macro conditions: six levels, excluded from the 43-feature count

| Feature | Formula | Source | Availability/lag rule | Unit |
| --- | --- | --- | --- | --- |
| risk_free_3m | DGS3MO / 100 | FRED/ALFRED DGS3MO | first vintage; next XNYS open assumption | annual decimal yield |
| treasury_10y | DGS10 / 100 | FRED/ALFRED DGS10 | first vintage; next XNYS open assumption | annual decimal yield |
| baa_yield | DBAA / 100 | FRED/ALFRED DBAA | first vintage; next XNYS open assumption | annual decimal yield |
| vix_level | VIXCLS | FRED/ALFRED VIXCLS | first vintage; next XNYS open assumption | index points |
| term_spread | (DGS10 - DGS3MO) / 100, matched observation date | FRED/ALFRED DGS10 and DGS3MO | first vintage; next XNYS open assumption | annual decimal yield spread |
| credit_spread | (DBAA - DGS10) / 100, matched observation date | FRED/ALFRED DBAA and DGS10 | first vintage; next XNYS open assumption | annual decimal yield spread |

These six conditions retain their raw units, are never cross-sectionally winsorized, filled or ranked, and are stored once per date in data/week2/macro_conditions.parquet. They are excluded from features_raw.parquet and features_processed.parquet. The processed feature file contains the 38 normalized firm characteristics and five conditional products, plus report-only missing indicators.

## Financial periods and shares

All financial access goes through storage.as_of at forecast_at, retaining the existing next-XNYS-open-after-filed-date convention. Only us-gaap and dei are accepted. The SEC frame/fy/fp fields do not select periods. USD facts are matched by start/end, tag priority, and available filing vintage. Direct single quarters are preferred; Q2/Q3 are reconstructed by adjacent same-fiscal-year YTD differences, and Q4 is annual minus nine-month YTD. TTM is a direct 330-400 day annual total or four adjacent 65-110 day quarters. No missing quarter is invented and no YTD observation is multiplied by four. Direct annual totals take priority over sums derived for the same year end. Gross profit can be aligned TTM revenue minus TTM cost of revenue. Stock values use the latest available period.

Year-ago comparisons require a matching fiscal endpoint within 30 calendar days of one year earlier. Average assets/equity are the arithmetic mean of the current and matching year-ago stock value. Period ends older than 550 calendar days are unavailable. Non-positive denominators produce missing ratios; income and cash-flow numerators may be negative. No financial level is filled before its availability date.

Shares prefer dei:EntityCommonStockSharesOutstanding, then us-gaap:CommonStockSharesOutstanding. Shares are adjusted by observed splits after the share measurement date and through the origin, using only available price rows. These are reported common shares outstanding, not verified free float. Concurrent tickers sharing one CIK do not receive issuer-shares-based market-cap/value features because class allocation is ambiguous. More subtle share-class limitations remain disclosed in the quality report. Turnover moves daily volume into the as-of share units, includes zero volume, and requires the same 80% observation rule. It does not cross a session with unknown split information.

## Market-series assumption (requires supervisor confirmation)

DGS3MO, DGS10, DBAA and VIXCLS use the earliest non-missing value present in the downloaded ALFRED archive for each observation date. Assumed available_at is the next XNYS session open strictly after that date. This is a declared market-data availability assumption: the earliest archived vintage is not independently verified as the true first publication. Original fred.py, the fred table and raw ALFRED timestamps are unchanged. T10Y3M and BAA10Y are unused. Spreads require the same observation date for both legs. The latest usable observation and calendar-day staleness are recorded for every monthly cutoff. Month-end-origin/forecast staleness is reported for all four levels and both spreads.

The user's 2010-2023 revision audit found multiple distinct values on 0.25% of DBAA dates, 0.11% of VIXCLS dates, 0.08% of DGS10 dates, and 0% of DGS3MO dates. The W2 runner regenerates its own revision counts. Week 7 will compare the declared assumption against original ALFRED availability, without rewriting the raw data.

## Cross-sectional transformation and interactions

Only the 38 firm characteristics are transformed within each historical CIK-mapped cohort/date: winsorize at the 1st/99th percentile, fill with that date's winsorized median, then map average ranks to [-1,1]. Constants and entirely missing firm columns map to neutral zero. Entirely missing firm feature/date combinations are recorded explicitly; a wholly unavailable OLS input blocks that month's benchmark. Median imputation does not restore a missing entry/exit price or a missing label.

Next multiply each already-normalized firm characteristic by its raw macro condition. Do not winsorize, fill or rank the products again. For example, normalized momentum of -1, 0 and 1 combined with term spread 0.01 produces -0.01, 0 and 0.01; term spread 0.04 produces -0.04, 0 and 0.04. Macro magnitude and sign are retained across months. The five products are conditional slopes, not five guaranteed independent signals. A missing macro condition propagates to its products without cross-sectional imputation.

Coverage for the 38 firm characteristics measures observed raw values before median imputation. Coverage for each interaction measures joint availability of the original firm input and raw macro condition, before firm imputation; it does not count a median-filled firm input as originally observed. The 43 model features are included in feature_quality.csv and the coverage heatmap. Coverage of the six macro conditions is exported separately in macro_condition_coverage.csv and summarized in the report. Outlier counts refer only to the winsorized firm inputs; products are not screened against new percentile bounds.

A coverage break is an absolute month-to-month change of at least 20 percentage points. Every row of coverage_breaks.csv is reproduced in the data-quality report body. If none are found, the report states that explicitly.

## Labels and evaluation

The label runs from entry-session adjusted_open to the 21st XNYS session adjusted_close, counting entry as session one. Subtract the origin-known DGS3MO annual decimal yield times actual elapsed calendar time (including the intraday fraction) / 365, with simple ACT/365 accrual. Cross-sectional label rank is the midpoint percentile (average_rank-0.5)/n. The middle third has a missing classification label; bottom/top thirds receive 0/1. Missing delisting or terminal prices remain missing and are reported.

The splitter records expanding training, a trailing 12-month validation block and one new test month at each monthly step. Overlapping training label intervals are purged. A full calendar month between label completion and the next evaluated block is an additional conservative embargo. Untuned OLS is refitted on all eligible mature pre-test labels, including past validation observations, subject to the same gap. The first OOS month is January 2016; 2015 calibration is reported separately. Dates from 2024 onward, including label windows crossing that boundary, are rejected without an explicit unlock flag. The W2 runner does not offer an unlock option.

## Primary references

- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- FRED/ALFRED API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- yfinance PriceHistory: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html

Coverage is generated from the user's database by the W2 command; no full-universe result is fabricated in this package.
