# When can monthly crime forecasts be trusted?

A forecasting study of recorded property and violent incidents in Montréal, 2015 to 2026.

**Author:** Dora Zhuang

Most forecasting write-ups report one accuracy number from one test period. This project asks a
harder question: across 77 forecast origins and five crime categories, **where do statistical
models add value over "same month last year", and how far can their uncertainty ranges be
trusted?**

## Findings

**1. Models beat the benchmark, but mostly at short horizons.**
Averaged over all categories, horizons and origins, ETS has a MASE of 1.01 against 1.25 for the
seasonal naive benchmark, an error about 19% lower. One month ahead the gap is large (0.75
against 1.25). Six months ahead it has nearly closed (1.13 against 1.23).

![MASE by horizon](figures/fig2_mase_by_horizon.png)

**2. A structural break defeats every method.**
Vehicle thefts rose from about 4,400 a year (2015 to 2020) to 11,500 in 2023, then fell back.
No model anticipated either turn. ETS had a MASE of 2.63 on 2021 forecast months, and only 40%
of outcomes fell inside its 80% interval.

![Vehicle theft forecasts](figures/fig4_vehicle_theft.png)

**3. Prediction intervals fail exactly when they matter.**
During 2020 and 2021, nominal 80% intervals from ETS covered 50% and 61% of outcomes. From 2023
onward they covered 78% to 81%. Interval quality depends on the period, so a single coverage
number hides the problem.

![Coverage by year](figures/fig3_coverage_by_year.png)

**4. Standard conformal calibration overcorrects after a shock.**
Recalibrating intervals with past forecast errors pushed 80% coverage up to 91%, because the
calibration history still contained the 2020 and 2021 errors. Adaptive conformal inference stayed
near the target (81%), at the cost of wider intervals than the model's own in calm periods.

![Calibration comparison](figures/fig5_calibration.png)

**5. At district level, splitting the city forecast beats forecasting each district.**
Extending the study to 29 police districts gives 180 series in four levels. The smallest series
(one category in one district) have a median of 14 incidents a month. Forecast independently,
the series do not add up: the sum of the district forecasts differs from the city forecast by
6% on average. Among five ways of making them consistent, the simplest one, splitting the city
total by each district's share of the last 12 months, is the most accurate at district level.

| Method | Total | Category | District | Category x district |
|---|---|---|---|---|
| Top-down (last-year shares) | 0.84 | 0.97 | **0.80** | **0.81** |
| WLS reconciliation | 0.88 | **0.92** | 0.81 | 0.85 |
| Independent ETS (does not add up) | 0.84 | 1.00 | 0.84 | 0.85 |
| Bottom-up | 0.94 | 0.98 | 0.83 | 0.85 |
| OLS reconciliation | 0.84 | 0.98 | 0.82 | 0.87 |
| Same month last year | 1.19 | 1.26 | 1.03 | 1.05 |

MASE pooled within each level, same 77 origins and horizons as above. No single method wins at
every level: bottom-up is clearly worst for the city total, and WLS is best for categories.

## Results in numbers

Rolling-origin backtest, origins December 2019 to April 2026, horizons 1 to 6 months.
MASE below 1 means the forecast beats the in-sample seasonal naive error.

| Category | ETS | ARIMA | Theta | Same month last year | Last month |
|---|---|---|---|---|---|
| Break and enter | 0.85 | 0.87 | 0.90 | 0.88 | 1.06 |
| Mischief | 0.93 | 1.04 | 0.99 | 1.13 | 1.55 |
| Robbery | 0.91 | 0.79 | 1.00 | 1.02 | 0.96 |
| Theft from vehicle | 0.91 | 0.95 | 0.92 | 1.12 | 1.34 |
| Vehicle theft | 1.46 | 1.42 | 1.40 | 2.09 | 1.45 |
| **All** | **1.01** | **1.02** | **1.04** | **1.25** | **1.27** |

80% interval calibration for ETS, origins December 2021 to April 2026 (the first two years are
needed to build a calibration set). The interval score rewards narrow intervals and penalizes
misses; lower is better.

| Method | Coverage (target 80%) | Interval score |
|---|---|---|
| Model interval | 78.7% | 3.77 |
| Conformal, all history | 91.2% | 4.46 |
| Conformal, last 36 errors | 87.2% | 4.35 |
| Adaptive conformal | 81.3% | 4.33 |

## Method

- **Data.** 365,404 incidents recorded by the Service de police de la Ville de Montréal (SPVM),
  January 2015 to October 2026, aggregated to citywide monthly counts for five categories.
- **Backtest.** At each of 77 origins every model is refit on the data available at that origin,
  then forecasts 1 to 6 months ahead with 80% and 95% intervals. This gives 2,310 forecasts per
  model.
- **Models.** ETS, ARIMA and Theta with automatic order selection, against two naive benchmarks.
- **Metrics.** MASE for point accuracy; empirical coverage and the interval (Winkler) score for
  uncertainty.
- **Hierarchy.** 5 categories by 29 districts (145 bottom series), plus district, category and
  city totals. Reconciliation is implemented from the projection formula in `src/hierarchy.py`:
  bottom-up, top-down, and OLS and structurally weighted (WLS) projections.
- **Calibration.** Split conformal on relative errors with an expanding and a 36-forecast
  window, and adaptive conformal inference (Gibbs and Candès, 2021) with a step size fixed in
  advance at 0.05. Every calibration set uses only outcomes observed before the forecast origin.

## Limitations

- The counts are incidents **recorded** by police. They reflect reporting and recording
  practice as well as underlying crime, and they are not a measure of neighbourhood risk.
- The SPVM states this dataset is not its official statistics. Incident types can be revised
  after the initial report, so the most recent months may still change.
- When one event involves several offences, only the most serious is published.
- Five districts were merged into neighbours during the period. Their history is mapped to
  today's boundaries. Two mergers are documented by the SPVM; the other three (11 into 9, 24
  into 26, 33 into 31) are inferred from the data and should be confirmed.
- District results are point forecasts only; intervals were not evaluated at that level.
- "Offences causing death" (two or three a month) is excluded as too sparse for these models.
- The adaptive conformal results start in December 2021, so that method was not tested through
  the 2020 shock itself.
- Forecasts here are an accuracy study, not a tool for deploying police resources.

## Reproduce

```bash
pip install -r requirements.txt
python src/download_data.py   # about 365,000 rows from the open data API
python src/backtest.py        # about 2 minutes on 4 cores
python src/calibrate.py
python src/hierarchy.py       # about 6 minutes on 4 cores
python src/figures.py
```

Results depend on the download date, because the source is updated daily.

## Next steps

- Prediction intervals for the district hierarchy.
- Count models and methods that adapt faster to level shifts.
- An explicit treatment of late revisions in recent months.

## Data source

[Actes criminels](https://donnees.montreal.ca/dataset/actes-criminels), Service de police de la
Ville de Montréal, via the Ville de Montréal open data portal. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Reference

Gibbs, I., and Candès, E. (2021). Adaptive conformal inference under distribution shift.
*Advances in Neural Information Processing Systems*, 34.
