The most difficult issue this week was determining when SEC filings could be used for forecasting. I considered using pre-market filings on the same day, or consolidating all filings for the next trading day. Ultimately, I chose the second option. The original plan stated "available from the next trading session," so a unified postponement to the next trading day is more conservative. Since forecasting is conducted after the close of the last trading day of each month, this rule only affects a small number of filings accepted pre-market on the last day of the month, having a minimal impact on the results, but the dates are easier to verify. I used the acceptance time from the SEC index page or file header as a basis, and then cross-checked it using Submissions JSON.



Another problem was that full processing took longer than I expected. My initial estimate, based on a three-year trial run with 20 companies, was 45.04 hours for SEC downloads, 46.27 hours for FinBERT scoring, and approximately 97 hours including sentence vectors. Later, I changed the download to four parallel threads, sending a maximum of eight requests per second, and then used an RTX 5060 Ti to score the text. Downloaded and scoring results are cached, allowing the process to resume after an interruption. After rerunning with a small sample size, the total time for all three full-scale estimations decreased to approximately 22 hours.



A total of 117,877 documents were scored, generating 65,788 monthly text features for each company. During the 95 out-of-sample months in the development phase, incorporating only sentiment changes improved the average Rank IC by 0.01118. The t-value for the paired test was 1.968, below the preset threshold. The formal determination of H2 will be postponed until the end of the storage period in week 7. Regarding the thematic stock pool, all three structural checks passed, but the small cross-section of approximately 35 stocks resulted in significant noise in the monthly IC; the predictive contraction implemented according to the preset rules did not improve the Rank IC (−0.00288 → −0.00721, t = −0.23).



If I were to redo it, I would have taken several pre-market, after-market, and holiday-prepared documents earlier to verify the trading days, and then used a small sample to test the download and scoring times separately. This time, after examining the two time periods separately, I realized that both downloading and scoring were slow, and simply replacing the GPU would only solve part of the problem.

