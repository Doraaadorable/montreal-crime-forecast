"""Does training on all series at once help, holding the model fixed?

The global model beats per-series ETS, but that comparison changes two things at
once: the model class and whether series are pooled. This script changes only one.
The same LightGBM settings and the same inputs are trained two ways:

    pooled:    one model per origin and horizon, fitted on all 145 series
    separate:  one model per origin, horizon and series, fitted on that series alone

Series identity is left out of both, since a single-series model cannot use it.

Usage:
    python src/pooling_test.py    # about 15 minutes on 4 cores
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from count_models import FEATURES, LGB_PARAMS, feature_panel, global_model_forecasts
from data import load_incidents, monthly_by_category_pdq
from hierarchy import HORIZON, MIN_TRAIN_MONTHS
from significance import block_bootstrap_ratio

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FEATURE_SETS = {
    "Recent levels only (y0, mean3, mean12)": ["y0", "mean3", "mean12"],
    "All inputs except series identity": [f for f in FEATURES if f not in ("category", "district")],
}


def separate_one_series(panels: dict[int, pd.DataFrame], origins, features: list[str]) -> pd.DataFrame:
    params = {**LGB_PARAMS, "n_jobs": 1}
    frames = []
    for h, panel in panels.items():
        for cutoff in origins:
            train = panel[panel["ds"] <= cutoff]
            test = panel[panel["t"] == cutoff]
            out = test[["unique_id", "ds", "target"]].rename(columns={"target": "y"})
            if train["target"].sum() == 0:  # a series with no incidents yet: forecast zero
                out["forecast"] = 0.0
            else:
                model = lgb.LGBMRegressor(**params).fit(train[features], train["target"])
                out["forecast"] = model.predict(test[features])
            out["cutoff"], out["horizon"] = cutoff, h
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def separate_model_forecasts(bottom: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    origins = bottom.index[MIN_TRAIN_MONTHS - 1: len(bottom) - HORIZON]
    panels = {h: feature_panel(bottom, h) for h in range(1, HORIZON + 1)}
    series = panels[1]["unique_id"].unique()
    jobs = (delayed(separate_one_series)(
        {h: p[p["unique_id"] == uid] for h, p in panels.items()}, origins, features)
        for uid in series)
    return pd.concat(Parallel(n_jobs=4)(jobs), ignore_index=True)


def main() -> None:
    bottom = monthly_by_category_pdq(load_incidents())
    scale = (pd.read_parquet(RESULTS / "hierarchy_forecasts.parquet")
             .query("level == 'Category x district'")[["unique_id", "cutoff", "scale"]]
             .drop_duplicates())
    rows = []
    for name, features in FEATURE_SETS.items():
        print(name)
        result = {}
        for how, fn in (("pooled", lambda: global_model_forecasts(bottom, features, verbose=False)),
                        ("separate", lambda: separate_model_forecasts(bottom, features))):
            f = fn().merge(scale, on=["unique_id", "cutoff"])
            f["abs_error"] = (f["y"] - f["forecast"]).abs()
            result[how] = f
            print(f"  {how}: MASE {f['abs_error'].sum() / f['scale'].sum():.3f}")
        # Score both on the same forecasts.
        keys = ["unique_id", "ds", "cutoff"]
        both = result["pooled"].merge(result["separate"], on=keys, suffixes=("_pooled", "_separate"))
        per_origin = both.groupby("cutoff")[["abs_error_pooled", "abs_error_separate"]].sum()
        ratio, lo, hi = block_bootstrap_ratio(per_origin["abs_error_pooled"].to_numpy(),
                                              per_origin["abs_error_separate"].to_numpy())
        total_scale = both["scale_pooled"].sum()
        rows.append({
            "inputs": name,
            "pooled_mase": round(both["abs_error_pooled"].sum() / total_scale, 3),
            "separate_mase": round(both["abs_error_separate"].sum() / total_scale, 3),
            "error_ratio_pooled_vs_separate": round(ratio, 3),
            "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "share_of_origins_pooled_wins": round(float(
                (per_origin["abs_error_pooled"] < per_origin["abs_error_separate"]).mean()), 2),
            "n_forecasts": len(both),
        })
    table = pd.DataFrame(rows).set_index("inputs")
    table.to_csv(RESULTS / "pooling_test.csv")
    pd.set_option("display.width", 220)
    print(table)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
