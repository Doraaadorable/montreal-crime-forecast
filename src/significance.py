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
    rec = pd.read_parquet(RESULTS / "hierarchy_forecasts.parquet")
    rec = rec[rec["level"] == "Category x district"]
    district = {m: g for m, g in rec.groupby("method")}
    district["Global"] = pd.read_parquet(RESULTS / "hierarchy_global_forecasts.parquet")
    ets_d, naive_d = "ETS base (not coherent)", "Same month last year"

    # (label, forecasts of method A, forecasts of method B, loss column)
    pairs: list = []

    def city(a: str, b: str, label: str, mask=lambda d: slice(None)) -> None:
        da, db = by_model[a], by_model[b]
        pairs.append((label, da[mask(da)], db[mask(db)], "abs_scaled_error"))

    city("AutoETS", "SeasonalNaive", "Citywide: ETS vs same month last year")
    city("AutoETS", "SeasonalNaive", "  1 month ahead", lambda d: d["horizon"] == 1)
    city("AutoETS", "SeasonalNaive", "  6 months ahead", lambda d: d["horizon"] == 6)
    for cat in sorted(long["unique_id"].unique()):
        city("AutoETS", "SeasonalNaive", f"  {cat}", lambda d, c=cat: d["unique_id"] == c)
    city("AutoARIMA", "AutoETS", "Citywide: ARIMA vs ETS")
    city("Ensemble", "AutoETS", "Citywide: ensemble vs ETS")
    city("Ensemble", "SeasonalNaive", "Citywide: ensemble vs same month last year")
    for a, b, label in [("Global", "Top-down", "District: global model vs top-down"),
                        ("Global", ets_d, "District: global model vs separate ETS"),
                        ("Top-down", ets_d, "District: top-down vs separate ETS"),
                        (ets_d, naive_d, "District: separate ETS vs same month last year")]:
        pairs.append((label, district[a], district[b], "abs_error"))

    rows: list = []
    robust = []
    last = long["ds"].max()

    def add(label: str, check: str, a: pd.Series, b: pd.Series, block: int = BLOCK) -> None:
        r, lo, hi = block_bootstrap_ratio(a.to_numpy(), b.to_numpy(), block=block)
        verdict = "A better" if hi < 1 else "B better" if lo > 1 else "no clear difference"
        robust.append({"comparison": label.strip(), "check": check, "error_ratio": round(r, 3),
                       "ci_low": round(lo, 3), "ci_high": round(hi, 3), "verdict": verdict})

    # Every comparison gets the same four checks.
    for label, fa, fb, loss in pairs:
        a, b = per_origin(fa, loss), per_origin(fb, loss)
        compare(rows, label, a, b)
        for block in (6, 12, 18):
            add(label, f"block of {block}", a, b, block)
        early = a.index <= "2022-12-31"
        add(label, "origins to 2022-12", a[early], b[early])
        add(label, "origins from 2023-01", a[~early], b[~early])
        # Are the last three months of the data driving the result? This drops their
        # scores only; it does not re-run the backtest with delayed data.
        cut = last - pd.DateOffset(months=3)
        add(label, "newest 3 target months not scored",
            per_origin(fa[fa["ds"] <= cut], loss), per_origin(fb[fb["ds"] <= cut], loss))

    robust = pd.DataFrame(robust)
    robust.to_csv(RESULTS / "significance_robustness.csv", index=False)
    stable = (robust.groupby("comparison", sort=False)["verdict"]
              .agg(lambda v: v.iloc[1] if v.nunique() == 1 else "changes with the check"))
    table = pd.DataFrame(rows).set_index("comparison")
    table["verdict"] = table["verdict"].replace("not distinguishable", "no clear difference")
    table["across_all_checks"] = stable.to_numpy()
    table.to_csv(RESULTS / "significance.csv")
    pd.set_option("display.width", 220)
    print(table[["error_ratio", "ci_low", "ci_high", "verdict", "across_all_checks"]])
    changed = robust[robust["comparison"].isin(stable[stable == "changes with the check"].index)]
    print("\nComparisons whose reading changes:\n", changed.to_string(index=False))
    ens = long.groupby("model")["abs_scaled_error"].mean().round(3).sort_values()
    print("\nCitywide MASE including the ensemble:\n", ens)


if __name__ == "__main__":
    main()
