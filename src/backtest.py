"""Rolling-origin backtest of monthly crime forecasts by category.

Every forecast origin refits each model on data available at that origin only,
then forecasts 1 to H months ahead with 80% and 95% prediction intervals.

Usage:
    python src/backtest.py        # writes results/backtest_forecasts.parquet + summary CSVs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

from data import load_incidents, monthly_by_category

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

HORIZON = 6
SEASON = 12
MIN_TRAIN_MONTHS = 60
LEVELS = [80, 95]
MODELS = ["Naive", "SeasonalNaive", "AutoETS", "AutoARIMA", "AutoTheta"]


def run_cross_validation(series: pd.DataFrame) -> pd.DataFrame:
    n_months = series["ds"].nunique()
    n_windows = n_months - MIN_TRAIN_MONTHS - HORIZON + 1
    sf = StatsForecast(
        models=[
            Naive(),
            SeasonalNaive(season_length=SEASON),
            AutoETS(season_length=SEASON),
            AutoARIMA(season_length=SEASON),
            AutoTheta(season_length=SEASON),
        ],
        freq="MS",
        n_jobs=-1,
    )
    cv = sf.cross_validation(
        df=series, h=HORIZON, step_size=1, n_windows=n_windows, level=LEVELS, refit=True
    )
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    cv["horizon"] = (
        (cv["ds"].dt.year - cv["cutoff"].dt.year) * 12 + cv["ds"].dt.month - cv["cutoff"].dt.month
    )
    return cv


def add_mase_scale(cv: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """In-sample seasonal-naive MAE at each origin: the denominator of MASE."""
    rows = []
    for uid, grp in series.groupby("unique_id"):
        y = grp.set_index("ds")["y"]
        abs_seasonal_diff = (y - y.shift(SEASON)).abs()
        for cutoff in cv.loc[cv["unique_id"] == uid, "cutoff"].unique():
            rows.append((uid, cutoff, abs_seasonal_diff.loc[:cutoff].mean()))
    scale = pd.DataFrame(rows, columns=["unique_id", "cutoff", "scale"])
    return cv.merge(scale, on=["unique_id", "cutoff"])


def to_long(cv: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for m in MODELS:
        part = cv[["unique_id", "ds", "cutoff", "horizon", "y", "scale"]].copy()
        part["model"] = m
        part["forecast"] = cv[m]
        for lv in LEVELS:
            part[f"lo{lv}"] = cv[f"{m}-lo-{lv}"]
            part[f"hi{lv}"] = cv[f"{m}-hi-{lv}"]
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["abs_scaled_error"] = (out["y"] - out["forecast"]).abs() / out["scale"]
    for lv in LEVELS:
        out[f"covered{lv}"] = out["y"].between(out[f"lo{lv}"], out[f"hi{lv}"])
        alpha = 1 - lv / 100
        width = out[f"hi{lv}"] - out[f"lo{lv}"]
        below = (out[f"lo{lv}"] - out["y"]).clip(lower=0)
        above = (out["y"] - out[f"hi{lv}"]).clip(lower=0)
        # Interval (Winkler) score, scaled like MASE so categories are comparable.
        out[f"interval_score{lv}"] = (width + 2 / alpha * (below + above)) / out["scale"]
    return out


def summarize(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        long.groupby(by)
        .agg(
            MASE=("abs_scaled_error", "mean"),
            coverage80=("covered80", "mean"),
            coverage95=("covered95", "mean"),
            interval_score80=("interval_score80", "mean"),
            n=("y", "size"),
        )
        .round(3)
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    series = monthly_by_category(load_incidents())
    print(f"{series['unique_id'].nunique()} series, "
          f"{series['ds'].min():%Y-%m} to {series['ds'].max():%Y-%m}")
    cv = add_mase_scale(run_cross_validation(series), series)
    long = to_long(cv)
    long.to_parquet(RESULTS / "backtest_forecasts.parquet", index=False)

    overall = summarize(long, ["model"]).sort_values("MASE")
    by_cat = summarize(long, ["unique_id", "model"])
    by_h = summarize(long, ["model", "horizon"])
    overall.to_csv(RESULTS / "summary_overall.csv")
    by_cat.to_csv(RESULTS / "summary_by_category.csv")
    by_h.to_csv(RESULTS / "summary_by_horizon.csv")

    pd.set_option("display.width", 200)
    print(f"\nOrigins: {long['cutoff'].min():%Y-%m} to {long['cutoff'].max():%Y-%m} "
          f"({long['cutoff'].nunique()} origins)")
    print("\nOverall:\n", overall)
    print("\nMASE by category:\n", by_cat["MASE"].unstack().round(3))
    print("\n80% interval coverage by category:\n", by_cat["coverage80"].unstack().round(3))
    print("\nMASE by horizon:\n", by_h["MASE"].unstack().round(3))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
