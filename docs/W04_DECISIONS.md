# Week 4 decisions

## Baseline and common development sample

- The source is the published Week 3 RF run `194b8ba55f8844e1b7057b44869f4685`. The required split SHA-256 is `cc60d7a8a5524628241cff0e898238a2d9867b63641435dd6934969a5a34c11e`.
- The 95 test months run from January 2016 to November 2023. Require mean monthly Rank IC 0.00394 at the five-decimal precision published in the Week 3 report. Check saved predictions first when present. Refit using the eight logged January selections, one fit per test month on the original refit rows. Do not tune.
- January 2024 onward remains sealed. The thematic analysis reuses the Week 3 nontext model as the transfer baseline. H2 and the text augmentations use the frozen S&P 500 stock-months.

## Filing availability, comparable filings and H2

- An accepted date is the calendar date in America/New_York. Open of the first XNYS session after that date is the earliest available timestamp, including pre-market acceptance. The code uses SEC header or filing-index time and records JSON disagreements. A conflict between authoritative clocks or a missing authoritative clock excludes the filing from features.
- Verify summer and winter wall clocks against SEC index pages and headers before downloading filing bodies. The three spot checks are Apple accessions `0000320193-23-000006`, `0000320193-23-000077`, and `0000320193-23-000106`. The index and header should agree on February 2 18:01:30, August 3 18:04:43, and November 2 18:08:27, respectively. The raw Submissions JSON values are retained for comparison.
- The pilot uses 20 continuously represented historical CIKs selected in CIK order and calendar entry months from 2021 through 2023. It checks the pipeline rather than estimating incremental value. The full corpus begins in 2010 and uses historical CIKs from the frozen sample.
- Keep primary 8-K item narratives, 10-K Items 7 and 1A, and 10-Q Part I Item 2 and Part II Item 1A. Common boilerplate, identifiable forward-looking and safe-harbor disclaimer paragraphs, and exact repeats within a document are removed. Attachments, exhibit-only reports, amendments and unparseable narratives are excluded. Narrative extraction is heuristic and coverage is reported.
- At each actual Week 3 forecast cutoff, keep separate groups for periodic filings (10-K/10-Q) and current reports (8-K). Within each group the latest usable filing can supply a level for 180 days. Changes use a newly available filing within that group against a prior filing of the same form. For 8-K, additionally require the same item signature and an earlier event date. For 10-Q, allow a 50–225 day fiscal-period gap. For 10-K, use the previous fiscal-year filing within 300–430 days. The latest comparable 10-K cosine similarity is carried for up to 430 days.
- Fit a separate monthly ridge head for each filing group. Within each group, its head scores both the current and prior comparable embeddings for the firm-month. The head uses one-month embargo, interval purging and labels already mature at that month's forecast cutoff. All historical training, validation and test feature rows follow that rule. The pretrained NLP models and dictionary are fixed for retrospective research. We cannot verify training-corpus availability for those pretrained assets at every historical forecast.
- H2 uses the RF with the six sentiment-change columns, three per filing group, and their raw missing flags. Support H2 only when the paired mean Rank IC increment is positive and the six-lag Newey–West t exceeds 2. An undefined t makes the result inconclusive. All-text and individual raw-signal tests are exploratory. Individual features use original scores on observed stock-months with no RF fits. Each is paired with the base forest on the same available rows. The all-text and H2 forests retain missing-text rows and use the same frozen test keys. Build flags from raw values, then apply the Week 2 monthly 1st and 99th percentile clipping, median imputation and rank normalization to [-1, 1] before appending numeric text columns to the Week 3 matrix. Partial rank correlations include 21-day reversal, size, book-to-market and 12-minus-1-month momentum controls as applicable.
- Download issuers in up to four worker threads, each with its own HTTP session. All threads share one rate limiter that every HTTP request passes through, including followed redirects and retries, capped at 8 SEC requests per second in total, below the SEC limit of 10. Each cache write uses its own temporary file. Scoring stays in the main thread in issuer order. NLP runs on the CUDA GPU when PyTorch can see one (`nlp_device: auto`) and otherwise on the CPU. The resolved device and batch size are part of the score-cache key, so CPU and GPU scores are never mixed. Threads, the rate cap and the device change runtime only. Filing selection, acceptance times, extraction and the sampled sentences are unchanged, and GPU and CPU scores differ only by floating-point rounding. Loughran–McDonald uses the full cleaned text. FinBERT and the sentence encoder share up to 128 sampled sentences per document by default. Sample without replacement with seed 1 and keep the selected sentences in document order. Short filings use all available sentences. Both models still split a long selected sentence into token windows when needed.
- The pilot prints SEC download, FinBERT and embedding estimates separately. Scale observed stage times by the ratio of planned full issuer-years to 60 pilot issuer-years. The SEC estimate combines pilot request counts with cold-request timing and the configured throttle floor. If the corpus was cached, use the refreshed SEC clock check to estimate the per-request cost. Time estimates exclude other work and depend on filing density, hardware and network conditions.
- Log each variant once per completed configuration and keep failed attempts in the cumulative MLflow count. Cost figures are optional estimates from configured compute cost per hour and measured scorer time.

SEC references

- [SEC EDGAR filing hours](https://www.sec.gov/submit-filings)
- [Apple February index](https://www.sec.gov/Archives/edgar/data/320193/000032019323000006/0000320193-23-000006-index.htm) and [header](https://www.sec.gov/Archives/edgar/data/320193/000032019323000006/0000320193-23-000006-index-headers.html)
- [Apple August index](https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/0000320193-23-000077-index.htm) and [header](https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/0000320193-23-000077-index-headers.html)
- [Apple November index](https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106-index.htm) and [header](https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106-index-headers.html)

## Thematic diagnosis and remedy selection

The starting list includes 53 candidates in robotics, industrial automation, motion control, machine vision, edge AI and industrial AI software. Forty-two US-listed equities and ADRs are eligible by security type. STM files Form 20-F and is classified separately as U.S.-listed foreign common equity. Every candidate has `needs_verification = true` pending full listing and eligibility review. The remaining 11 stay in the coverage table. Within eligible names, use forecast-known prices and dated listing boundaries. The first observed vendor quote supplies a conservative usable-history start when an exact listing date is unavailable. The curated list is a retrospective screen, and data gaps for delisted firms can produce survivorship bias. Missing XBRL data are measured and eligible price-only rows receive the Week 3 imputation policy.

Run the existing frozen RF model on the thematic panel, then estimate the S&P 500 minus thematic Rank IC gap on matching months. Include cross-section counts, first quote and documented listing dates, short histories, fundamental coverage, volatility, S&P 500 membership overlap, and a seeded equally sized S&P 500 subsample comparison.

| Structural check | Pass condition |
| --- | --- |
| Thin cross-sections | At most 20% of test months have fewer than 15 scored names |
| Short listing histories | At most 25% of eligible test stock-months have fewer than 252 prior quote sessions |
| Fundamental coverage | At least 60% of eligible fundamental feature cells are observed |

A failed or unavailable structural check selects `pooled`. Pooled training keeps each overlapping stock-month once, adds a thematic indicator and uses only eligible mature labels across both samples. If all three checks pass, select `shrinkage`. Combine thematic-only and transferred cross-sectional percentile predictions. The thematic weight is the number of mature thematic training rows divided by that count plus 2,000.

The initial S&P 500 versus thematic IC gap and its Newey–West test are reported for diagnosis. Running `apply` always uses the structural rule to select and evaluate one remedy. The before and after comparison uses the same thematic stock-months and reports matched mean Rank IC, the paired change and its six-lag Newey–West test. These results are exploratory.

Selected listing sources

- [Arm Nasdaq listing](https://www.nasdaq.com/press-release/arm-announces-closing-of-initial-public-offering-and-full-exercise-of-underwriters)
- [ABB delisting](https://www.sec.gov/Archives/edgar/data/1091587/000110465924026633/abb20231231.htm)
- [Symbotic Nasdaq debut](https://ir.symbotic.com/news-releases/news-release-details/symbotic-debuts-nasdaq-under-ticker-sym)
- [Ingersoll Rand ticker change](https://www.sec.gov/Archives/edgar/data/1699150/000114036120004816/form8k.htm)
- [C3.ai initial trading date](https://ir.c3.ai/static-files/0923dd61-7186-4880-b44c-dc7c155fdfad)
- [AutoStore Oslo listing](https://www.euronext.com/en/about/media/euronext-press-releases/autostore-lists-oslo-bors)
- [STMicroelectronics Form 20-F and NYSE common-share listing](https://www.sec.gov/Archives/edgar/data/932787/000162828024006353/R1.htm)
