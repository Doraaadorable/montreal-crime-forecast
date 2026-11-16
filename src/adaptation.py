"""Would a shorter memory have coped with the vehicle-theft surge?

ETS fitted on the full history was slow to follow the 2021 to 2023 surge. A model
that only sees recent data should adapt faster, at the price of noisier estimates
when nothing is changing. This script refits ETS at every origin on the last 24,
36 or 60 months only and compares it with the full-history fit on the same
forecasts.

Usage:
    python src/adaptation.py      # needs results/backtest_forecasts.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from backtest import HORIZON, MIN_TRAIN_MONTHS, SEASON
from data import load_incidents, monthly_by_category
from significance import block_bootstrap_ratio

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WINDOWS = [24, 36, 60]
SURGE = ("2021-01-01", "2023-12-31")  # forecast months during the rise and the peak


def windowed_forecasts(series: pd.DataFrame, window: int) -> pd.DataFrame:
    n_windows = series["ds"].nunique() - MIN_TRAIN_MONTHS - HORIZON + 1
    sf = StatsForecast(models=[AutoETS(season_length=SEASON)], freq="MS", n_jobs=-1)
    cv = sf.cross_validation(df=series, h=HORIZON, step_size=1, n_windows=n_windows,
                             input_size=window, level=[80], refit=True)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    return cv.rename(columns={"AutoETS": "forecast", "AutoETS-lo-80": "lo80",
                              "AutoETS-hi-80": "hi80"}).assign(memory=f"last {window} months")


def main() -> None:
    series = monthly_by_category(load_incidents())
    full = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    full = full[full["model"] == "AutoETS"].assign(memory="full history")
    keys = ["unique_id", "ds", "cutoff"]
    frames = [full[keys + ["y", "forecast", "lo80", "hi80", "scale", "memory"]]]
    for window in WINDOWS:
        w = windowed_forecasts(series, window).merge(full[keys + ["scale"]], on=keys)
        frames.append(w[keys + ["y", "forecast", "lo80", "hi80", "scale", "memory"]])
    f = pd.concat(frames, ignore_index=True)
    f["abs_scaled_error"] = (f["y"] - f["forecast"]).abs() / f["scale"]
    f["covered80"] = f["y"].between(f["lo80"], f["hi80"])
    f["period"] = np.where(f["ds"].between(*SURGE), "2021 to 2023", "other years")
    f["series"] = np.where(f["unique_id"] == "Vehicle theft", "Vehicle theft", "Other four categories")
    f.to_parquet(RESULTS / "adaptation_forecasts.parquet", index=False)

    order = ["full history"] + [f"last {w} months" for w in WINDOWS]
    table = (f.groupby(["series", "period", "memory"])
             .agg(MASE=("abs_scaled_error", "mean"), coverage80=("covered80", "mean"))
             .round(3).unstack("memory"))
    table = table.reindex(columns=order, level=1)
    table.to_csv(RESULTS / "adaptation_by_memory.csv")
    pd.set_option("display.width", 220)
    print(table.to_string())

    rows = []
    for (series_name, period), g in f.groupby(["series", "period"]):
        base = g[g["memory"] == "full history"].groupby("cutoff")["abs_scaled_error"].sum()
        for window in WINDOWS:
            short = g[g["memory"] == f"last {window} months"].groupby("cutoff")["abs_scaled_error"].sum()
            ratio, lo, hi = block_bootstrap_ratio(short.to_numpy(), base.to_numpy())
            rows.append({"series": series_name, "period": period, "memory": f"last {window} months",
                         "error_ratio_vs_full_history": round(ratio, 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    ratios = pd.DataFrame(rows)
    ratios.to_csv(RESULTS / "adaptation_ratios.csv", index=False)
    print("\n", ratios.to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
