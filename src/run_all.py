"""Run the whole study in order. About 30 minutes on 4 cores.

Usage:
    python src/run_all.py                  # download, then every analysis step
    python src/run_all.py --skip-download  # reuse data/raw/actes_criminels.csv
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "download_data.py",   # incident records from the open data API
    "backtest.py",        # citywide rolling-origin forecasts
    "calibrate.py",       # conformal recalibration of intervals
    "hierarchy.py",       # district hierarchy and reconciliation
    "count_models.py",    # pooled Poisson model, count intervals, ablation
    "pooling_test.py",    # pooled vs separate training, same model
    "significance.py",    # bootstrap intervals and robustness checks
    "adaptation.py",      # short-memory ETS
    "planning.py",        # capacity plans
    "monitoring.py",      # CUSUM alarms on forecast errors
    "figures.py",
    "build_page.py",
]

if __name__ == "__main__":
    steps = STEPS[1:] if "--skip-download" in sys.argv else STEPS
    for step in steps:
        start = time.time()
        print(f"== {step}", flush=True)
        subprocess.run([sys.executable, str(SRC / step)], check=True)
        print(f"   done in {time.time() - start:.0f}s", flush=True)
