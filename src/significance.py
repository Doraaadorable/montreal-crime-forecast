"""How sure are the accuracy comparisons?

Each comparison is an error ratio (method A / method B) with a 95% interval from a
moving-block bootstrap over the 77 forecast origins. Blocks of 12 consecutive
origins are resampled together because errors at neighbouring origins overlap and
are strongly correlated; resampling origins one by one would overstate certainty.

Also adds a simple ensemble: the average of the ETS, ARIMA and Theta forecasts.

Usage:
    python src/significance.py    # needs the backtest, hierarchy and count-model results
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BLOCK = 12
N_BOOT = 5000
ENSEMBLE_MEMBERS = ["AutoETS", "AutoARIMA", "AutoTheta"]


def block_bootstrap_ratio(loss_a: np.ndarray, loss_b: np.ndarray, block: int = BLOCK,
                          n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of total losses with a 95% moving-block bootstrap interval.

    loss_a and loss_b hold one total loss per forecast origin, in time order.
    """
    n = len(loss_a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    ratios = loss_a[idx].sum(axis=1) / loss_b[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(loss_a.sum() / loss_b.sum()), float(lo), float(hi)


def per_origin(df: pd.DataFrame, loss: str) -> pd.Series:
    return df.groupby("cutoff")[loss].sum().sort_index()


def compare(rows: list, label: str, a: pd.Series, b: pd.Series) -> None:
    ratio, lo, hi = block_bootstrap_ratio(a.to_numpy(), b.to_numpy())
    verdict = "A better" if hi < 1 else "B better" if lo > 1 else "not distinguishable"
    rows.append({"comparison": label, "error_ratio": round(ratio, 3),
                 "ci_low": round(lo, 3), "ci_high": round(hi, 3), "verdict": verdict,
                 "share_of_origins_A_wins": round(float((a < b).mean()), 2)})


def add_ensemble(long: pd.DataFrame) -> pd.DataFrame:
    members = long[long["model"].isin(ENSEMBLE_MEMBERS)]
    keys = ["unique_id", "ds", "cutoff", "horizon"]
    ens = members.groupby(keys, as_index=False).agg(
        y=("y", "first"), scale=("scale", "first"), forecast=("forecast", "mean"))
    ens["model"] = "Ensemble"
    ens["abs_scaled_error"] = (ens["y"] - ens["forecast"]).abs() / ens["scale"]
    return pd.concat([long, ens], ignore_index=True)


def main() -> None:
    long = add_ensemble(pd.read_parquet(RESULTS / "backtest_forecasts.parquet"))
    by_model = {m: g for m, g in long.groupby("model")}
    loss = "abs_scaled_error"
    rows: list = []

    def city(a: str, b: str, label: str, mask=lambda d: slice(None)) -> None:
        da, db = by_model[a], by_model[b]
        compare(rows, label, per_origin(da[mask(da)], loss), per_origin(db[mask(db)], loss))

    city("AutoETS", "SeasonalNaive", "Citywide: ETS vs same month last year")
    city("AutoETS", "SeasonalNaive", "  1 month ahead", lambda d: d["horizon"] == 1)
    city("AutoETS", "SeasonalNaive", "  6 months ahead", lambda d: d["horizon"] == 6)
    for cat in sorted(long["unique_id"].unique()):
        city("AutoETS", "SeasonalNaive", f"  {cat}", lambda d, c=cat: d["unique_id"] == c)
    city("AutoARIMA", "AutoETS", "Citywide: ARIMA vs ETS")
    city("Ensemble", "AutoETS", "Citywide: ensemble vs ETS")
    city("Ensemble", "SeasonalNaive", "Citywide: ensemble vs same month last year")

    rec = pd.read_parquet(RESULTS / "hierarchy_forecasts.parquet")
    rec = rec[rec["level"] == "Category x district"]
    gbm = pd.read_parquet(RESULTS / "hierarchy_global_forecasts.parquet")
    bottom = {m: per_origin(g, "abs_error") for m, g in rec.groupby("method")}
    bottom["Global"] = per_origin(gbm, "abs_error")
    compare(rows, "District: global model vs top-down", bottom["Global"], bottom["Top-down"])
    compare(rows, "District: global model vs separate ETS", bottom["Global"],
            bottom["ETS base (not coherent)"])
    compare(rows, "District: top-down vs separate ETS", bottom["Top-down"],
            bottom["ETS base (not coherent)"])
    compare(rows, "District: separate ETS vs same month last year",
            bottom["ETS base (not coherent)"], bottom["Same month last year"])

    table = pd.DataFrame(rows).set_index("comparison")
    table.to_csv(RESULTS / "significance.csv")
    pd.set_option("display.width", 200)
    print(table)
    ens = long.groupby("model")["abs_scaled_error"].mean().round(3).sort_values()
    print("\nCitywide MASE including the ensemble:\n", ens)
    print("\nEnsemble MASE by category:\n",
          long[long["model"].isin(["Ensemble", "AutoETS"])]
          .groupby(["unique_id", "model"])["abs_scaled_error"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
