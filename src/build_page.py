"""Build the interactive summary page from the backtest results.

Reads site/page.html, embeds the numbers it charts, and writes docs/index.html
(served by GitHub Pages).

Usage:
    python src/build_page.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw" / "actes_criminels.csv"

DISTRICT_ROWS = {
    "Global Poisson boosting": "One model for all series (Poisson boosting)",
    "Top-down": "Split the city forecast by last-year shares",
    "ETS base (not coherent)": "Separate ETS model per series",
    "Same month last year": "Same month last year",
}
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n')


def series_payload(long: pd.DataFrame) -> dict:
    out: dict = {}
    ets = long[long["model"] == "AutoETS"]
    naive = long[long["model"] == "SeasonalNaive"]
    for cat, by_cat in ets.groupby("unique_id"):
        out[cat] = {}
        for h, g in by_cat.groupby("horizon"):
            g = g.sort_values("ds")
            sn = naive[(naive["unique_id"] == cat) & (naive["horizon"] == h)]
            out[cat][int(h)] = {
                "ds": g["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": g["y"].round().astype(int).tolist(),
                "f": g["forecast"].round(1).tolist(),
                "lo": g["lo80"].round(1).tolist(),
                "hi": g["hi80"].round(1).tolist(),
                "mase": round(float(g["abs_scaled_error"].mean()), 3),
                "mase_sn": round(float(sn["abs_scaled_error"].mean()), 3),
            }
    return out


def main() -> None:
    long = pd.read_parquet(RESULTS / "backtest_forecasts.parquet")
    ets = long[long["model"] == "AutoETS"]
    coverage = ets.groupby(ets["ds"].dt.year)["covered80"].mean()
    district = pd.read_csv(RESULTS / "bottom_point_accuracy.csv", index_col=0)
    downloaded = datetime.fromtimestamp(RAW.stat().st_mtime, tz=timezone.utc)
    data = {
        "series": series_payload(long),
        "coverage": {"year": [int(y) for y in coverage.index],
                     "ets": [round(float(v), 3) for v in coverage]},
        "district": {
            "columns": ["All horizons", "1 month", "3 months", "6 months"],
            "rows": [{"label": label,
                      "values": [round(float(district.loc[key, c]) + 1e-9, 2)
                                 for c in ["All", "h1", "h3", "h6"]]}
                     for key, label in DISTRICT_ROWS.items()],
        },
        "downloaded": f"{downloaded:%B} {downloaded.day}, {downloaded:%Y}",
    }
    template = (ROOT / "site" / "page.html").read_text(encoding="utf-8")
    body = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(HEAD + body + "\n</body>\n</html>\n", encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").touch()
    (ROOT / "site" / "_artifact.html").write_text(body, encoding="utf-8")
    print(f"Wrote docs/index.html ({len(body) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
