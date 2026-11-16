# When can monthly crime forecasts be trusted?

Between 2020 and 2023, vehicle thefts in Montréal more than doubled, then fell 43% in two
years. None of the seven methods tested caught either turn. This project replays six years of forecasts on
365,000 police-recorded incidents to find where forecasts hold up and where they break.

**Dora Zhuang** ·
[Interactive summary](https://doraaadorable.github.io/montreal-crime-forecast/) ·
[Full findings](docs/FINDINGS.md)

## Findings

Every model is refit at 77 points in time and forecasts 1 to 6 months ahead, using only the
months before that point.

**1. Models beat "same month last year" by 19%, mostly in the short run.**
One month ahead, ETS cuts the benchmark's error by 40%. Six months ahead the gain is 8%, and
since 2023 it cannot be told from zero.

**2. A structural break defeats every method tested.**
On 2021 forecast months, ETS errors on vehicle theft were 2.6 times the benchmark scale, and
only 40% of outcomes fell inside its 80% interval. Fitting on only the last two or three years
did not help (error ratio 0.98, interval 0.95 to 1.05) and raised errors on the other four
categories by 25% to 45%. A standard monitor on the forecast errors
first signalled in April 2021 and again in July, when thefts had reached 652 a month against
392 in 2020. The peak of 1,205 came in August 2023.

![Vehicle theft forecasts](figures/fig4_vehicle_theft.png)

**3. Uncertainty ranges fail when they matter.**
80% intervals covered 50% of outcomes in 2020 and 61% in 2021, then 78% to 81% from 2023 on.
The conformal settings tested did not fix this. The standard version overcorrects to 91% because its
history still holds the shock. The adaptive version averages 81% only by swinging from 92% in
2022 to 73% in 2026, and the model's own interval scored better in every year.

![Coverage by year](figures/fig3_coverage_by_year.png)

**4. For small districts, pooling beats separate models.**
A typical district series has 14 incidents a month. Splitting the citywide forecast by each
district's share of the last year beats a model per district by 5%. Pooling is what
helps: with the model and its inputs held fixed, training on all 145 series at once cuts error
by 7% against training each series alone (ratio 0.93, 95% interval 0.91 to 0.95).

**5. Plans built on the 90th percentile fall short more often than intended.**
One month ahead, such a plan should be short in 10% of months. It was short in 14.5% on
average: 21% for vehicle theft, 10% for robbery and mischief, and 25% of months in 2021. Under
an illustrative cost of nine to one for shortfall against spare capacity, and across all
horizons, it costs 29% less than "last year plus 20%" and 6% less than a flat 20% margin on
the same forecast.

## Which differences are real

Error ratios with 95% intervals from a block bootstrap over the 77 forecast origins. Below 1
favours the first method. An interval that includes 1 means the two cannot be separated.

| Comparison | Error ratio | 95% interval |
|---|---|---|
| Citywide: ETS vs same month last year | 0.81 | 0.71 to 0.86 |
| Same, 1 month ahead | 0.60 | 0.51 to 0.66 |
| Same, 6 months ahead | 0.92 | 0.81 to 0.99 |
| Citywide: ARIMA vs ETS | 1.00 | 0.95 to 1.09 |
| Citywide: average of ETS, ARIMA, Theta vs ETS | 0.97 | 0.95 to 1.00 |
| District: top-down split vs separate ETS | 0.95 | 0.93 to 0.97 |
| District: one pooled model vs top-down split | 0.99 | 0.98 to 1.00 |

No clear difference was detected between ETS and ARIMA, or between a pooled gradient-boosting
model and the top-down split.

All 13 comparisons in the [full table](docs/FINDINGS.md) were rerun with bootstrap blocks of 6
and 18, on each half of the test period, and without scoring the newest three months. Ten
keep the same reading. Three change with the check: the six-month gain, the forecast average
against ETS, and the pooled model against the top-down split.

## Method

- **Data.** 365,404 incidents recorded by the Service de police de la Ville de Montréal (SPVM),
  January 2015 to October 2026, in five categories and 29 police districts. Downloaded 15
  November 2026; the file hash is in `data/manifest.json`.
- **Backtest.** 77 rolling origins, horizons of 1 to 6 months, 80% and 95% intervals.
- **Models.** ETS, ARIMA, Theta, their average, a pooled LightGBM model with a Poisson loss,
  and two naive benchmarks. Hyperparameters were fixed in advance.
- **Uncertainty.** Coverage and interval scores; split and adaptive conformal calibration;
  Poisson and negative binomial intervals for small counts.
- **Hierarchy.** City, category, district and category by district (180 series), reconciled
  bottom-up, top-down, and by OLS and WLS projection, written out in `src/hierarchy.py`.
- **Significance.** Moving-block bootstrap on error ratios, in blocks of 12 origins.

The counts are incidents recorded by police, so they reflect reporting and recording practice
as well as crime itself. Past months come from today's version of the data, which may differ
from what was published at the time.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_all.py                   # downloads today's data, then runs every step
pytest                                  # method checks, no data needed
```

The source is updated daily, so a new download gives slightly different numbers. To reproduce
the committed results, use the data version in `data/manifest.json`: place that file at
`data/raw/actes_criminels.csv`, then

```bash
python src/run_all.py --skip-download   # about 22 minutes on 4 cores
python src/verify_outputs.py            # compares 40 outputs with results/output_hashes.sha256
```

With that data file and the environment in `results/environment.txt`, a run from a fresh clone
reproduced every committed result, figure and the summary page byte for byte. Other package
versions may change trailing digits or image encoding.

## Data and license

Data: [Actes criminels](https://donnees.montreal.ca/dataset/actes-criminels), Service de police
de la Ville de Montréal, Ville de Montréal open data portal,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: [MIT](LICENSE).
