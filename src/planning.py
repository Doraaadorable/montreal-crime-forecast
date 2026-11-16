"""What the forecasts mean for someone who has to plan capacity.

Suppose monthly workload scales with recorded incidents, and running short costs
more than holding spare capacity. A planner who accepts being short one month in
ten should plan to the 90th percentile of the forecast, which is the upper end of
the 80% interval. This script scores that plan against simpler rules:

    shortfall rate   share of months where incidents exceeded the plan (target 10%)
    spare capacity   average amount planned above what happened, as % of actual
    cost             shortfall weighted 9 to 1 against spare capacity (pinball loss
                     at 0.9), relative to the seasonal-naive scale; lower is better

Usage:
    python src/planning.py        # needs results/backtest_forecasts.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGET = 0.9
SHOCK_YEARS = [2020, 2021, 2022]


def score(d: pd.DataFrame, plan: pd.Series) -> dict:
    short = (d["y"] - plan).clip(lower=0)
    spare = (plan - d["y"]).clip(lower=0)
    return {
        "shortfall_rate": (d["y"] > plan).mean(),
        "spare_capacity_pct": spare.sum() / d["y"].sum() * 100,
        "cost": ((TARGET * short + (1 - TARGET) * spare) / d["scale"]).mean(),
    }


def main() -> None:
    long = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    ets = long[long["model"] == "AutoETS"].reset_index(drop=True)
    naive = long[long["model"] == "SeasonalNaive"].reset_index(drop=True)
    last_year = naive["forecast"]  # same month last year, aligned row by row with ets
    assert (ets[["unique_id", "ds", "cutoff"]].to_numpy()
            == naive[["unique_id", "ds", "cutoff"]].to_numpy()).all()

    plans = {
        "ETS 90th percentile": ets["hi80"],
        "ETS forecast, no margin": ets["forecast"],
        "ETS forecast + 10%": ets["forecast"] * 1.10,
        "ETS forecast + 20%": ets["forecast"] * 1.20,
        "Last year's count + 10%": last_year * 1.10,
        "Last year's count + 20%": last_year * 1.20,
    }
    ets["year"] = ets["ds"].dt.year
    periods = {"All years": ets["year"] > 0,
               "2020 to 2022": ets["year"].isin(SHOCK_YEARS),
               "2023 to 2026": ~ets["year"].isin(SHOCK_YEARS)}
    rows = []
    for period, mask in periods.items():
        for name, plan in plans.items():
            rows.append({"period": period, "plan": name, **score(ets[mask], plan[mask])})
    table = pd.DataFrame(rows).set_index(["period", "plan"]).round(3)
    table.to_csv(RESULTS / "planning.csv")
    pd.set_option("display.width", 200)
    print(table.to_string())

    by_cat = pd.DataFrame({
        cat: score(ets[ets["unique_id"] == cat], ets.loc[ets["unique_id"] == cat, "hi80"])
        for cat in sorted(ets["unique_id"].unique())}).T.round(3)
    by_cat.to_csv(RESULTS / "planning_by_category.csv")
    print("\nETS 90th percentile plan by category:\n", by_cat.to_string())

    by_year = ets.assign(short=ets["y"] > ets["hi80"]).groupby("year")["short"].mean().round(3)
    by_year.to_csv(RESULTS / "planning_shortfall_by_year.csv")
    print("\nShortfall rate of the ETS 90th percentile plan by year (target 0.10):\n",
          by_year.to_string())


if __name__ == "__main__":
    main()
