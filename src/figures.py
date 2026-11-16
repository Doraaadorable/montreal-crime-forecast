"""Render the README figures from the backtest and calibration results.

Usage:
    python src/figures.py         # writes figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import load_incidents, monthly_by_category

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
MODEL_COLORS = {
    "AutoETS": "#2a78d6",
    "AutoARIMA": "#eb6834",
    "AutoTheta": "#1baf7a",
    "SeasonalNaive": "#eda100",
    "Naive": "#e87ba4",
}
MODEL_LABELS = {
    "AutoETS": "ETS",
    "AutoARIMA": "ARIMA",
    "AutoTheta": "Theta",
    "SeasonalNaive": "Same month last year",
    "Naive": "Last month",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_series(series: pd.DataFrame) -> None:
    cats = sorted(series["unique_id"].unique())
    fig, axes = plt.subplots(len(cats), 1, figsize=(9, 9), sharex=True)
    for ax, cat in zip(axes, cats):
        s = series[series["unique_id"] == cat]
        ax.plot(s["ds"], s["y"], color=MODEL_COLORS["AutoETS"], lw=1.6)
        ax.set_title(cat)
        ax.set_ylim(0, s["y"].max() * 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].text(0, 1.32, "Recorded incidents per month, Montréal, 2015 to 2026",
                 transform=axes[0].transAxes, fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig1_monthly_series.png")


def fig_mase_by_horizon(long: pd.DataFrame) -> None:
    table = long.groupby(["model", "horizon"])["abs_scaled_error"].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for model, color in MODEL_COLORS.items():
        ax.plot(table.index, table[model], color=color, lw=2, marker="o", ms=5,
                label=MODEL_LABELS[model])
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Model advantage shrinks as the horizon grows")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_mase_by_horizon.png")


def fig_coverage_by_year(long: pd.DataFrame) -> None:
    models = ["AutoETS", "AutoARIMA", "SeasonalNaive"]
    table = (long.assign(year=long["ds"].dt.year)
             .groupby(["year", "model"])["covered80"].mean().unstack("model"))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(0.8, color=MUTED, lw=1, ls="--")
    ax.text(table.index.min(), 0.81, "nominal 80%", color=MUTED, va="bottom", fontsize=9)
    for model in models:
        ax.plot(table.index, table[model], color=MODEL_COLORS[model], lw=2, marker="o", ms=5,
                label=MODEL_LABELS[model])
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(table.index)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Year of the forecast month")
    ax.set_ylabel("Share of outcomes inside the 80% interval")
    ax.set_title("Model intervals were too narrow during 2020 to 2022, then recovered")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig3_coverage_by_year.png")


def fig_vehicle_theft(long: pd.DataFrame, horizon: int = 3) -> None:
    f = long[(long["unique_id"] == "Vehicle theft") & (long["model"] == "AutoETS")
             & (long["horizon"] == horizon)].sort_values("ds")
    color = MODEL_COLORS["AutoETS"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(f["ds"], f["lo80"], f["hi80"], color=color, alpha=0.18, lw=0,
                    label="80% interval")
    ax.plot(f["ds"], f["forecast"], color=color, lw=2, label=f"ETS forecast, {horizon} months ahead")
    ax.plot(f["ds"], f["y"], color=INK, lw=1.6, label="Recorded")
    ax.set_ylim(0, None)
    ax.set_ylabel("Vehicle thefts per month")
    ax.set_title("Vehicle theft: forecasts lag the 2021 to 2023 surge and its reversal")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "fig4_vehicle_theft.png")


def fig_calibration(cal: pd.DataFrame) -> None:
    methods = {
        "covered80": ("Model interval", "#2a78d6"),
        "covered80_expanding": ("Conformal, all history", "#eb6834"),
        "covered80_rolling36": ("Conformal, last 36", "#1baf7a"),
        "covered80_aci": ("Adaptive conformal", "#eda100"),
    }
    ets = cal[cal["model"] == "AutoETS"]
    table = ets.groupby("unique_id")[list(methods)].mean().sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvline(0.8, color=MUTED, lw=1, ls="--")
    ax.text(0.797, len(table) - 0.45, "nominal 80%", color=MUTED, ha="right", fontsize=9)
    for col, (label, color) in methods.items():
        ax.scatter(table[col], range(len(table)), s=70, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(table)), table.index)
    ax.set_ylim(-0.6, len(table) - 0.2)
    ax.set_xlim(0.7, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of outcomes inside the 80% interval (origins 2021-12 to 2026-04)")
    ax.set_title("Average coverage by calibration method (yearly swings are in the table)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig5_calibration.png")


def main() -> None:
    long = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    cal = pd.read_parquet(RESULTS / "calibrated_forecasts.parquet")
    fig_series(monthly_by_category(load_incidents()))
    fig_mase_by_horizon(long)
    fig_coverage_by_year(long)
    fig_vehicle_theft(long)
    fig_calibration(cal)
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
