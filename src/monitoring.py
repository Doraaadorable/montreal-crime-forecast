"""If a break cannot be forecast, how quickly can it be detected?

A CUSUM chart on one-month-ahead forecast errors. Each month the error is divided
by the standard deviation the model itself claimed, and deviations beyond a small
allowance accumulate. A sustained run of errors in one direction raises an alarm;
isolated misses do not. The chart uses only forecasts already scored, so every
alarm date is one a monitor would have seen at the time.

Settings are the textbook defaults (allowance 0.5, threshold 4, in standard
deviations) and were not tuned on these series.

Usage:
    python src/monitoring.py      # needs results/backtest_forecasts.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALLOWANCE = 0.5
THRESHOLD = 4.0
Z80 = 1.2816  # half-width of an 80% normal interval in standard deviations


def cusum_alarms(z: pd.Series) -> pd.DataFrame:
    """Two-sided CUSUM. Both sums reset after an alarm."""
    up = down = 0.0
    rows = []
    for month, value in z.items():
        up = max(0.0, up + value - ALLOWANCE)
        down = max(0.0, down - value - ALLOWANCE)
        if up > THRESHOLD or down > THRESHOLD:
            rows.append({"month": month, "direction": "above forecast" if up > THRESHOLD
                         else "below forecast"})
            up = down = 0.0
    return pd.DataFrame(rows)


def main() -> None:
    long = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    one = long[(long["model"] == "AutoETS") & (long["horizon"] == 1)].copy()
    sigma = (one["hi80"] - one["lo80"]) / (2 * Z80)
    one["z"] = (one["y"] - one["forecast"]) / sigma

    alarms = []
    for cat, g in one.groupby("unique_id"):
        g = g.set_index("ds").sort_index()
        a = cusum_alarms(g["z"])
        if len(a):
            a["category"] = cat
            a["recorded"] = g.loc[a["month"], "y"].to_numpy()
            a["forecast"] = g.loc[a["month"], "forecast"].round().to_numpy()
            alarms.append(a)
    alarms = pd.concat(alarms, ignore_index=True)[["category", "month", "direction",
                                                   "recorded", "forecast"]]
    alarms["month"] = alarms["month"].dt.strftime("%Y-%m")
    alarms.to_csv(RESULTS / "monitoring_alarms.csv", index=False)
    print(f"Monitored {one['ds'].min():%Y-%m} to {one['ds'].max():%Y-%m}, "
          f"{one['ds'].nunique()} months per category\n")
    print(alarms.to_string(index=False))

    summary = alarms.groupby("category").size().rename("alarms").to_frame()
    summary["by_year"] = alarms.assign(year=alarms["month"].str[:4]).groupby("category")["year"].agg(
        lambda s: ", ".join(f"{y} ({n})" for y, n in s.value_counts().sort_index().items()))
    print("\n", summary.to_string())

    # Vehicle theft in detail. Months are the months the incidents occurred in; a monitor
    # would see each one only once that month's data was published.
    theft = one[one["unique_id"] == "Vehicle theft"].set_index("ds").sort_index()
    peak_month = theft["y"].idxmax()
    print(f"\nVehicle theft: peak {theft['y'].max():.0f} in {peak_month:%Y-%m}; "
          f"2020 monthly mean {theft.loc['2020', 'y'].mean():.0f}")
    print(alarms[alarms["category"] == "Vehicle theft"].to_string(index=False))

    # No downward alarm followed the peak. Is that because the model's stated
    # uncertainty had grown? Re-run the chart after the peak with the error scaled as
    # if uncertainty had stayed at its 2020 share of the forecast level.
    after = theft.loc[peak_month + pd.DateOffset(months=1):]
    sigma_all = (theft["hi80"] - theft["lo80"]) / (2 * Z80)
    share_2020 = (sigma_all / theft["forecast"]).loc["2020"].mean()
    share_after = (sigma_all / theft["forecast"]).loc[after.index].mean()
    rescaled = (after["y"] - after["forecast"]) / (share_2020 * after["forecast"])
    downward = lambda z: [m for m, d in zip(*[cusum_alarms(z).get(c, []) for c in ("month", "direction")])
                          if d == "below forecast"]
    negative_run = int((after["y"] < after["forecast"]).cummin().sum())
    check = pd.Series({
        "stated_sd_share_of_forecast_2020": round(share_2020, 3),
        "stated_sd_share_of_forecast_after_peak": round(share_after, 3),
        "downward_alarms_after_peak": len(downward(after["z"])),
        "downward_alarms_after_peak_with_2020_sd_share": len(downward(rescaled)),
        "consecutive_months_below_forecast_after_peak": negative_run,
    })
    check.to_csv(RESULTS / "monitoring_decline_check.csv", header=["value"])
    print("\nDecline after the peak:\n", check.to_string())

if __name__ == "__main__":
    main()
