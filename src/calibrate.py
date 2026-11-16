"""Conformal recalibration of prediction intervals, using past backtest errors only.

For a forecast made at origin c for horizon h, the calibration set is the set of
earlier forecasts of the same series, model and horizon whose target month had
already been observed at c. The interval is forecast * (1 +/- q), where q is the
conformal quantile of past relative absolute errors. Nothing from the future of
origin c is used, so the reported coverage is out-of-sample.

Usage:
    python src/calibrate.py       # needs results/backtest_forecasts.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LEVELS = [80, 95]
MIN_CALIBRATION = 24
WINDOWS = {"expanding": None, "rolling36": 36}
METHODS = [*WINDOWS, "aci"]
ACI_GAMMA = 0.05  # step size of adaptive conformal inference, fixed in advance


def conformal_quantile(scores: np.ndarray, level: int) -> float:
    n = len(scores)
    rank = min(int(np.ceil((n + 1) * level / 100)), n)
    return float(np.sort(scores)[rank - 1])


def level_quantile(scores: np.ndarray, miscoverage: float) -> float:
    """Quantile for a target miscoverage rate that ACI may push outside (0, 1).

    Gibbs and Candes return an infinite interval when the rate falls to 0 or below.
    An infinite interval cannot be scored or used, so this returns the largest error in
    the calibration window instead; a rate of 1 or more gives a zero-width interval.
    How often either happens is reported in results/aci_bounds.csv.
    """
    if miscoverage <= 0:
        return float(scores.max())
    if miscoverage >= 1:
        return 0.0
    rank = min(int(np.ceil((len(scores) + 1) * (1 - miscoverage))), len(scores))
    return float(np.sort(scores)[rank - 1])


def calibrate_group(grp: pd.DataFrame) -> pd.DataFrame:
    """grp: one series x model x horizon, sorted by cutoff."""
    grp = grp.sort_values("cutoff").reset_index(drop=True)
    score = ((grp["y"] - grp["forecast"]).abs() / grp["forecast"].clip(lower=1)).to_numpy()
    ds = grp["ds"].to_numpy()
    cutoff = grp["cutoff"].to_numpy()
    out = {f"q{lv}_{name}": np.full(len(grp), np.nan) for lv in LEVELS for name in METHODS}
    out.update({f"aci_rate{lv}": np.full(len(grp), np.nan) for lv in LEVELS})
    for i in range(len(grp)):
        resolved = ds[:i] <= cutoff[i]
        available = score[:i][resolved]
        if len(available) < MIN_CALIBRATION:
            continue
        # Adaptive conformal inference (Gibbs & Candes, 2021): widen after misses,
        # narrow after hits, using only forecasts whose outcome is already known.
        for lv in LEVELS:
            alpha = 1 - lv / 100
            past_q = out[f"q{lv}_aci"][:i][resolved]
            past_score = available[~np.isnan(past_q)]
            misses = past_score > past_q[~np.isnan(past_q)]
            alpha_t = alpha + ACI_GAMMA * np.sum(alpha - misses)
            out[f"aci_rate{lv}"][i] = alpha_t
            out[f"q{lv}_aci"][i] = level_quantile(available[-WINDOWS["rolling36"]:], alpha_t)
        for name, window in WINDOWS.items():
            cal = available if window is None else available[-window:]
            for lv in LEVELS:
                out[f"q{lv}_{name}"][i] = conformal_quantile(cal, lv)
    return grp.assign(**out)


def main() -> None:
    long = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    cal = pd.concat(
        [calibrate_group(g) for _, g in long.groupby(["unique_id", "model", "horizon"])],
        ignore_index=True,
    )
    # Compare raw and conformal intervals on the same forecasts only.
    cal = cal.dropna(subset=["q80_expanding"]).copy()
    for name in METHODS:
        for lv in LEVELS:
            half = cal["forecast"].clip(lower=1) * cal[f"q{lv}_{name}"]
            lo, hi = cal["forecast"] - half, cal["forecast"] + half
            cal[f"lo{lv}_{name}"], cal[f"hi{lv}_{name}"] = lo, hi
            cal[f"covered{lv}_{name}"] = cal["y"].between(lo, hi)
            alpha = 1 - lv / 100
            miss = (lo - cal["y"]).clip(lower=0) + (cal["y"] - hi).clip(lower=0)
            cal[f"interval_score{lv}_{name}"] = (hi - lo + 2 / alpha * miss) / cal["scale"]
    cal.to_parquet(RESULTS / "calibrated_forecasts.parquet", index=False)

    def table(by: list[str]) -> pd.DataFrame:
        cols = {
            "raw_cov80": ("covered80", "mean"),
            "expanding_cov80": ("covered80_expanding", "mean"),
            "rolling36_cov80": ("covered80_rolling36", "mean"),
            "aci_cov80": ("covered80_aci", "mean"),
            "raw_cov95": ("covered95", "mean"),
            "expanding_cov95": ("covered95_expanding", "mean"),
            "rolling36_cov95": ("covered95_rolling36", "mean"),
            "aci_cov95": ("covered95_aci", "mean"),
            "raw_score80": ("interval_score80", "mean"),
            "expanding_score80": ("interval_score80_expanding", "mean"),
            "rolling36_score80": ("interval_score80_rolling36", "mean"),
            "aci_score80": ("interval_score80_aci", "mean"),
            "n": ("y", "size"),
        }
        return cal.groupby(by).agg(**cols).round(3)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Evaluated origins: {cal['cutoff'].min():%Y-%m} to {cal['cutoff'].max():%Y-%m}")
    by_model = table(["model"])
    by_cat = table(["unique_id", "model"])
    by_model.to_csv(RESULTS / "calibration_by_model.csv")
    by_cat.to_csv(RESULTS / "calibration_by_category.csv")
    print("\nBy model:\n", by_model)
    best = by_cat.xs("AutoETS", level="model")
    print("\nAutoETS by category:\n", best)

    bounds = pd.DataFrame({
        f"{lv}%": {"rate at or below 0 (capped at largest past error)":
                   (cal[f"aci_rate{lv}"] <= 0).mean(),
                   "rate at or above 1 (zero width)": (cal[f"aci_rate{lv}"] >= 1).mean()}
        for lv in LEVELS}).round(4)
    bounds.to_csv(RESULTS / "aci_bounds.csv")
    print("\nShare of adaptive intervals at a bound:\n", bounds)

    # Coverage, width and score together, by year, for the model interval and ACI.
    ets = cal[cal["model"] == "AutoETS"].assign(year=lambda d: d["ds"].dt.year)
    ets["width_raw"] = (ets["hi80"] - ets["lo80"]) / ets["scale"]
    ets["width_aci"] = (ets["hi80_aci"] - ets["lo80_aci"]) / ets["scale"]
    by_year_cal = ets.groupby("year").agg(
        model_coverage=("covered80", "mean"), aci_coverage=("covered80_aci", "mean"),
        model_width=("width_raw", "mean"), aci_width=("width_aci", "mean"),
        model_score=("interval_score80", "mean"), aci_score=("interval_score80_aci", "mean"),
        n=("y", "size")).round(3)
    by_year_cal.to_csv(RESULTS / "calibration_by_year.csv")
    print("\nETS 80% interval, model vs adaptive conformal, by target year:\n", by_year_cal)
    by_h = ets.groupby("horizon").agg(
        model_coverage=("covered80", "mean"), aci_coverage=("covered80_aci", "mean"),
        model_score=("interval_score80", "mean"), aci_score=("interval_score80_aci", "mean")).round(3)
    by_h.to_csv(RESULTS / "calibration_by_horizon.csv")
    print("\nBy horizon:\n", by_h)

    # Where raw model intervals fail: coverage of all backtest forecasts by target year.
    by_year = (
        long.assign(year=long["ds"].dt.year)
        .groupby(["year", "model"])["covered80"].mean().round(2).unstack("model")
    )
    by_year.to_csv(RESULTS / "raw_coverage_by_year.csv")
    print("\nRaw 80% interval coverage by target year (all categories):\n", by_year)

    # Accuracy and coverage by target year, to see the vehicle-theft regime shift.
    full = long[long["model"].isin(["AutoETS", "SeasonalNaive"])].copy()
    full["year"] = full["ds"].dt.year
    regime = (
        full.groupby(["unique_id", "year", "model"])
        .agg(MASE=("abs_scaled_error", "mean"), cov80=("covered80", "mean"))
        .round(2)
        .unstack("model")
    )
    regime.to_csv(RESULTS / "accuracy_by_year.csv")
    print("\nVehicle theft by target year:\n", regime.loc["Vehicle theft"])


if __name__ == "__main__":
    main()
