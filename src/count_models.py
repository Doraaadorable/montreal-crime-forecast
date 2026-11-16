"""Count forecasts for the 145 category x district series.

Two questions for the sparsest level of the hierarchy (median 14 incidents a month):

1. Point accuracy. Does one global model trained on all series at once (gradient
   boosting with a Poisson loss) beat per-series ETS and the top-down split?
2. Uncertainty. ETS intervals assume a continuous, symmetric error. Are intervals
   from a count distribution (Poisson, negative binomial) better calibrated?

Everything is rolling-origin and causal: a model for origin c and horizon h is
trained only on targets observed up to c, and the negative binomial dispersion at
origin c is estimated only from forecasts whose outcome was known at c.

Usage:
    python src/count_models.py    # needs results from src/hierarchy.py
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from data import load_incidents, monthly_by_category_pdq
from hierarchy import HORIZON, MIN_TRAIN_MONTHS, SEASON, level_mase

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LEVEL = 80
MIN_DISPERSION_SAMPLE = 500
# Fixed in advance; not tuned on the backtest.
LGB_PARAMS = dict(objective="poisson", n_estimators=200, learning_rate=0.05, num_leaves=15,
                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=42, n_jobs=4, verbose=-1)
FEATURES = ["y0", "y1", "y2", "mean3", "mean6", "mean12", "same_month_last_year",
            "category_momentum", "target_month", "category", "district"]


def feature_panel(bottom: pd.DataFrame, h: int) -> pd.DataFrame:
    """One row per series and month t, with features known at t and the target at t+h."""
    cats = bottom.columns.get_level_values(0)
    cat_total = bottom.T.groupby(level=0).sum().T
    momentum = (cat_total.rolling(3).mean() / cat_total.rolling(12).mean())[cats]
    momentum.columns = bottom.columns
    parts = {
        "y0": bottom, "y1": bottom.shift(1), "y2": bottom.shift(2),
        "mean3": bottom.rolling(3).mean(), "mean6": bottom.rolling(6).mean(),
        "mean12": bottom.rolling(12).mean(),
        "same_month_last_year": bottom.shift(SEASON - h),
        "category_momentum": momentum,
        "target": bottom.shift(-h),
    }
    panel = pd.concat({k: v.stack([0, 1], future_stack=True) for k, v in parts.items()}, axis=1)
    panel.index.names = ["t", "category", "district"]
    panel = panel.reset_index()
    panel["ds"] = panel["t"] + pd.DateOffset(months=h)
    panel["target_month"] = panel["ds"].dt.month
    panel["unique_id"] = panel["category"] + " | PDQ " + panel["district"].astype(str)
    panel["category"] = panel["category"].astype("category")
    panel["district"] = panel["district"].astype("category")
    return panel.dropna(subset=["mean12", "same_month_last_year"])


def global_model_forecasts(bottom: pd.DataFrame) -> pd.DataFrame:
    origins = bottom.index[MIN_TRAIN_MONTHS - 1: len(bottom) - HORIZON]
    frames = []
    for h in range(1, HORIZON + 1):
        panel = feature_panel(bottom, h)
        for cutoff in origins:
            train = panel[panel["ds"] <= cutoff]
            test = panel[panel["t"] == cutoff]
            model = lgb.LGBMRegressor(**LGB_PARAMS).fit(train[FEATURES], train["target"])
            out = test[["unique_id", "ds", "target"]].rename(columns={"target": "y"})
            out["forecast"] = model.predict(test[FEATURES])
            out["cutoff"], out["horizon"] = cutoff, h
            frames.append(out)
        print(f"  horizon {h} done")
    return pd.concat(frames, ignore_index=True)


def nb_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments alpha in Var(y) = mu + alpha * mu^2."""
    mu = np.clip(mu, 0.1, None)
    return float(max(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2), 0.0))


def count_intervals(f: pd.DataFrame) -> pd.DataFrame:
    """Add Poisson and negative binomial 80% intervals around the point forecast."""
    f = f.sort_values(["cutoff", "unique_id", "horizon"]).reset_index(drop=True)
    mu = np.clip(f["forecast"].to_numpy(), 0.05, None)
    lo_q, hi_q = (1 - LEVEL / 100) / 2, 1 - (1 - LEVEL / 100) / 2
    f["lo_poisson"], f["hi_poisson"] = stats.poisson.ppf(lo_q, mu), stats.poisson.ppf(hi_q, mu)
    f["lo_negbin"], f["hi_negbin"], f["alpha"] = np.nan, np.nan, np.nan
    for (cutoff, h), idx in f.groupby(["cutoff", "horizon"]).groups.items():
        past = f[(f["horizon"] == h) & (f["ds"] <= cutoff)]
        if len(past) < MIN_DISPERSION_SAMPLE:
            continue
        alpha = nb_dispersion(past["y"].to_numpy(), past["forecast"].to_numpy())
        m = mu[idx]
        if alpha < 1e-6:
            lo, hi = stats.poisson.ppf(lo_q, m), stats.poisson.ppf(hi_q, m)
        else:
            n, p = 1 / alpha, 1 / (1 + alpha * m)
            lo, hi = stats.nbinom.ppf(lo_q, n, p), stats.nbinom.ppf(hi_q, n, p)
        f.loc[idx, ["lo_negbin", "hi_negbin", "alpha"]] = np.column_stack(
            [lo, hi, np.full(len(m), alpha)])
    return f


def interval_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Coverage, width and pooled scaled interval score for named (lo, hi) columns."""
    alpha = 1 - LEVEL / 100
    rows = []
    for name, d in frames.items():
        miss = (d["lo"] - d["y"]).clip(lower=0) + (d["y"] - d["hi"]).clip(lower=0)
        score = (d["hi"] - d["lo"]) + 2 / alpha * miss
        rows.append({
            "method": name,
            "coverage80": d["y"].between(d["lo"], d["hi"]).mean(),
            "mean_width": (d["hi"] - d["lo"]).mean(),
            "interval_score": score.sum() / d["scale"].sum(),
            "n": len(d),
        })
    return pd.DataFrame(rows).set_index("method").round(3)


def main() -> None:
    bottom = monthly_by_category_pdq(load_incidents())
    series_mean = bottom.mean()
    series_mean.index = [f"{c} | PDQ {p}" for c, p in series_mean.index]

    print("Training global Poisson model (6 horizons x 77 origins)")
    gbm = global_model_forecasts(bottom)

    rec = pd.read_parquet(RESULTS / "hierarchy_forecasts.parquet")
    rec = rec[rec["level"] == "Category x district"]
    scale = rec[["unique_id", "cutoff", "scale"]].drop_duplicates()
    gbm = gbm.merge(scale, on=["unique_id", "cutoff"])
    gbm["abs_error"] = (gbm["y"] - gbm["forecast"]).abs()
    gbm["method"] = "Global Poisson boosting"

    # 1. Point accuracy at the bottom level.
    keep = ["Top-down", "ETS base (not coherent)", "Same month last year"]
    cols = ["unique_id", "ds", "cutoff", "horizon", "y", "forecast", "scale", "abs_error", "method"]
    points = pd.concat([rec[rec["method"].isin(keep)][cols], gbm[cols]], ignore_index=True)
    points["volume"] = pd.cut(
        points["unique_id"].map(series_mean), [0, 5, 20, np.inf],
        labels=["under 5 a month", "5 to 20", "over 20"])
    overall = level_mase(points, ["method"]).rename("All").to_frame()
    by_h = level_mase(points, ["method", "horizon"]).unstack().add_prefix("h")
    by_vol = level_mase(points, ["method", "volume"]).unstack()
    point_table = pd.concat([overall, by_h, by_vol], axis=1).sort_values("All").round(3)
    point_table.to_csv(RESULTS / "bottom_point_accuracy.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\nBottom-level MASE (pooled):\n", point_table)
    print("\nSeries per volume band:\n",
          points.drop_duplicates("unique_id")["volume"].value_counts().sort_index())

    # 2. Interval calibration at the bottom level, on forecasts all methods can score.
    gbm = count_intervals(gbm)
    top = rec[rec["method"] == "Top-down"][cols].copy()
    top = count_intervals(top)
    ets = pd.read_parquet(RESULTS / "hierarchy_base_forecasts.parquet")
    ets = ets.rename(columns={"AutoETS-lo-80": "lo", "AutoETS-hi-80": "hi"})
    ets["lo"] = ets["lo"].clip(lower=0)
    ets = ets.merge(scale, on=["unique_id", "cutoff"])[["unique_id", "ds", "cutoff", "y", "lo", "hi", "scale"]]

    scored = gbm.dropna(subset=["lo_negbin"])[["unique_id", "ds", "cutoff"]]

    def pick(d: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
        d = d.merge(scored, on=["unique_id", "ds", "cutoff"])
        return d.rename(columns={lo: "lo", hi: "hi"})[["y", "lo", "hi", "scale"]]

    intervals = interval_table({
        "ETS (normal interval)": pick(ets, "lo", "hi"),
        "Global boosting + Poisson": pick(gbm, "lo_poisson", "hi_poisson"),
        "Global boosting + negative binomial": pick(gbm, "lo_negbin", "hi_negbin"),
        "Top-down + Poisson": pick(top, "lo_poisson", "hi_poisson"),
        "Top-down + negative binomial": pick(top, "lo_negbin", "hi_negbin"),
    })
    intervals.to_csv(RESULTS / "bottom_interval_calibration.csv")
    print(f"\n80% intervals at the bottom level "
          f"(origins from {scored['cutoff'].min():%Y-%m}):\n", intervals)
    print("\nEstimated dispersion alpha (median over origins):",
          round(gbm["alpha"].median(), 3), "| top-down:", round(top["alpha"].median(), 3))
    gbm.to_parquet(RESULTS / "hierarchy_global_forecasts.parquet", index=False)


if __name__ == "__main__":
    main()
