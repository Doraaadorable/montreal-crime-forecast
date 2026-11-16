"""Hierarchical forecasts: city total, category, police district, category x district.

Base forecasts are produced independently for all 180 series in a rolling-origin
backtest, then reconciled so that they add up across the hierarchy. Reconciliation
is implemented directly from the projection formula
    y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat
so the method is explicit rather than hidden in a library call.

Usage:
    python src/hierarchy.py       # writes results/hierarchy_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from data import load_incidents, monthly_by_category_pdq

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

HORIZON = 6
SEASON = 12
MIN_TRAIN_MONTHS = 60
LEVELS = ["Total", "Category", "District", "Category x district"]


def build_hierarchy(bottom: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return all series (wide), the summing matrix S and each series' level."""
    cats = bottom.columns.get_level_values(0)
    pdqs = bottom.columns.get_level_values(1)
    bottom_ids = [f"{c} | PDQ {p}" for c, p in zip(cats, pdqs)]
    rows, ids, level = [np.ones(len(bottom_ids))], ["Total"], ["Total"]
    for c in sorted(set(cats)):
        rows.append((cats == c).astype(float)); ids.append(c); level.append("Category")
    for p in sorted(set(pdqs)):
        rows.append((pdqs == p).astype(float)); ids.append(f"PDQ {p}"); level.append("District")
    rows.extend(np.eye(len(bottom_ids))); ids.extend(bottom_ids)
    level.extend(["Category x district"] * len(bottom_ids))
    S = pd.DataFrame(np.vstack(rows), index=ids, columns=bottom_ids)
    all_series = pd.DataFrame(bottom.to_numpy() @ S.to_numpy().T, index=bottom.index, columns=ids)
    return all_series, S, pd.Series(level, index=ids, name="level")


def base_forecasts(all_series: pd.DataFrame) -> pd.DataFrame:
    long = all_series.rename_axis(index="ds", columns="unique_id").stack().rename("y").reset_index()
    long = long[["unique_id", "ds", "y"]]
    n_windows = len(all_series) - MIN_TRAIN_MONTHS - HORIZON + 1
    sf = StatsForecast(
        models=[AutoETS(season_length=SEASON), SeasonalNaive(season_length=SEASON)],
        freq="MS", n_jobs=-1,
    )
    cv = sf.cross_validation(df=long, h=HORIZON, step_size=1, n_windows=n_windows, refit=True)
    return cv.reset_index() if "unique_id" not in cv.columns else cv


def projection(S: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Map base forecasts of all series to coherent forecasts, for diagonal W."""
    w_inv = 1.0 / weights
    G = np.linalg.solve(S.T @ (w_inv[:, None] * S), S.T * w_inv)
    return S @ G


def reconcile(cv: pd.DataFrame, all_series: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    ids, n_bottom = list(S.index), S.shape[1]
    Sm = S.to_numpy()
    bottom_up = Sm @ np.hstack([np.zeros((n_bottom, len(ids) - n_bottom)), np.eye(n_bottom)])
    ols = projection(Sm, np.ones(len(ids)))
    wls = projection(Sm, Sm.sum(axis=1))  # weight = number of bottom series aggregated
    frames = []
    for cutoff, grp in cv.groupby("cutoff"):
        base = grp.pivot(index="unique_id", columns="ds", values="AutoETS").loc[ids].to_numpy()
        # Top-down: split the total forecast by each bottom series' share of the last year.
        last_year = all_series.loc[:cutoff].iloc[-SEASON:]
        shares = last_year[S.columns].sum() / last_year["Total"].sum()
        top_down = Sm @ np.outer(shares.to_numpy(), base[0])
        methods = {
            "ETS base (not coherent)": base,
            "Bottom-up": bottom_up @ base,
            "Top-down": top_down,
            "OLS reconciliation": ols @ base,
            "WLS reconciliation": wls @ base,
            "Same month last year": grp.pivot(index="unique_id", columns="ds",
                                              values="SeasonalNaive").loc[ids].to_numpy(),
        }
        actual = grp.pivot(index="unique_id", columns="ds", values="y").loc[ids]
        for name, values in methods.items():
            f = pd.DataFrame(np.clip(values, 0, None), index=ids, columns=actual.columns)
            out = f.stack().rename("forecast").reset_index()
            out.columns = ["unique_id", "ds", "forecast"]
            out["y"] = actual.stack().to_numpy()
            out["cutoff"], out["method"] = cutoff, name
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def add_scale(rec: pd.DataFrame, all_series: pd.DataFrame) -> pd.DataFrame:
    """In-sample seasonal-naive MAE per series and origin (the MASE denominator)."""
    abs_diff = (all_series - all_series.shift(SEASON)).abs()
    scale = abs_diff.expanding().mean().rename_axis(index="cutoff", columns="unique_id")
    scale = scale.stack().rename("scale").reset_index()
    return rec.merge(scale, on=["cutoff", "unique_id"])


def level_mase(rec: pd.DataFrame, by: list[str]) -> pd.Series:
    """Total absolute error over total seasonal-naive scale within each group.

    Pooling before dividing keeps sparse series (some district x category pairs
    are zero for years, so their own scale is zero) from producing infinite ratios.
    """
    sums = rec.groupby(by)[["abs_error", "scale"]].sum()
    return sums["abs_error"] / sums["scale"]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    bottom = monthly_by_category_pdq(load_incidents())
    all_series, S, level = build_hierarchy(bottom)
    print(f"{len(all_series.columns)} series ({S.shape[1]} bottom), {len(all_series)} months")
    print("Median monthly count by level:\n",
          all_series.median().groupby(level).median().reindex(LEVELS).round(1))

    cv = base_forecasts(all_series)
    rec = add_scale(reconcile(cv, all_series, S), all_series)
    rec["level"] = rec["unique_id"].map(level)
    rec["horizon"] = ((rec["ds"].dt.year - rec["cutoff"].dt.year) * 12
                      + rec["ds"].dt.month - rec["cutoff"].dt.month)
    rec["abs_error"] = (rec["y"] - rec["forecast"]).abs()
    rec.to_parquet(RESULTS / "hierarchy_forecasts.parquet", index=False)

    by_level = level_mase(rec, ["method", "level"]).unstack()[LEVELS]
    by_level = by_level.sort_values("Category x district").round(3)
    by_level.to_csv(RESULTS / "hierarchy_mase_by_level.csv")
    by_horizon = level_mase(rec[rec["level"] == "Category x district"],
                            ["method", "horizon"]).unstack().round(3)
    by_horizon.to_csv(RESULTS / "hierarchy_bottom_mase_by_horizon.csv")
    pd.set_option("display.width", 200)
    print("\nMASE by level and method:\n", by_level)
    print("\nBottom-level MASE by horizon:\n", by_horizon)

    # Coherence check: how far do the independent base forecasts miss adding up?
    base = rec[rec["method"] == "ETS base (not coherent)"]
    total = base[base["unique_id"] == "Total"].set_index(["cutoff", "ds"])["forecast"]
    summed = base[base["level"] == "Category x district"].groupby(["cutoff", "ds"])["forecast"].sum()
    gap = ((summed - total).abs() / total)
    print(f"\nBase forecasts: sum of bottom vs total differs by {gap.mean():.1%} on average "
          f"(max {gap.max():.1%})")
    pd.Series({"mean_gap": gap.mean(), "max_gap": gap.max()}).to_csv(
        RESULTS / "hierarchy_coherence_gap.csv", header=["value"])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
