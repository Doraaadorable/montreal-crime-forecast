"""Compare freshly generated outputs with the committed ones, after a full rerun.

Used by .github/workflows/reproduce.yml, which deletes every committed output, reruns the
whole study and then runs this script. Each output gets one of these verdicts:

    identical         same bytes as the committed file
    within tolerance  tables only: same shape and labels, numbers equal within a relative
                      tolerance of 1e-4 and an absolute one of 1e-6 (or, for rounded CSVs,
                      within one unit of the last printed decimal). Forecasts are stored in
                      single precision and come from numerical optimisers, so another
                      machine can move them in the fifth significant digit; a change in a
                      result is far larger than that
    differs           a table with a larger difference, missing columns or rows
    regenerated       figures and the HTML page that are not byte-identical; a new CPU or
                      library build can change image encoding without changing the data
    missing           the rerun did not write the file

The check fails on "differs" or "missing". Byte-identity of every file is reported but not
required, because floating-point results can vary slightly across machines.

Usage:
    python src/reproduce_check.py                # compare with HEAD, write the reports
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from verify_outputs import tracked_outputs

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reproduce_report.md"
RTOL = 1e-4
ATOL = 1e-6


def committed(path: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True,
                          capture_output=True).stdout


def last_decimal_unit(frame: pd.DataFrame) -> float:
    """One unit of the finest decimal printed in a CSV, e.g. 0.001 for values like 0.812."""
    places = 0
    for col in frame.select_dtypes("number"):
        text = frame[col].dropna().astype(str)
        frac = text.str.split(".").str[1].dropna()
        if len(frac):
            places = max(places, int(frac.str.len().max()))
    return 10.0 ** -places


def compare_tables(new: pd.DataFrame, old: pd.DataFrame, rounded: bool) -> tuple[str, str]:
    if list(new.columns) != list(old.columns) or new.shape != old.shape:
        return "differs", f"shape {old.shape} -> {new.shape}"
    num = old.select_dtypes("number").columns
    other = [c for c in old.columns if c not in num]
    for c in other:
        if not new[c].astype(str).equals(old[c].astype(str)):
            return "differs", f"labels changed in column {c!r}"
    if len(num) == 0:
        return "within tolerance", "no numeric columns"
    a = new[num].to_numpy(dtype=float)
    b = old[num].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return "differs", "missing values changed"
    diff = np.nan_to_num(np.abs(a - b))
    if np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
        return "within tolerance", f"max abs difference {diff.max():.2e}"
    if rounded:
        unit = last_decimal_unit(old)
        if diff.max() <= unit * 1.0001:
            n = int((diff > ATOL).sum())
            return "within tolerance", f"{n} values differ by one unit of the last decimal ({unit:g})"
    return "differs", f"max abs difference {diff.max():.3g}"


def check(path: str) -> tuple[str, str]:
    new_path = ROOT / path
    if not new_path.exists():
        return "missing", "not written by the rerun"
    old_bytes = committed(path)
    new_bytes = new_path.read_bytes()
    if hashlib.sha256(new_bytes).digest() == hashlib.sha256(old_bytes).digest():
        return "identical", ""
    if path.endswith(".csv"):
        new = pd.read_csv(io.BytesIO(new_bytes))
        old = pd.read_csv(io.BytesIO(old_bytes))
        return compare_tables(new, old, rounded=True)
    if path.endswith(".parquet"):
        new = pd.read_parquet(io.BytesIO(new_bytes))
        old = pd.read_parquet(io.BytesIO(old_bytes))
        return compare_tables(new, old, rounded=False)
    return "regenerated", "not byte-identical (image or page encoding)"


def main() -> int:
    rows = [(path, *check(path)) for path in tracked_outputs()]
    counts = pd.Series([r[1] for r in rows]).value_counts()
    failed = [r for r in rows if r[1] in ("differs", "missing")]
    lines = ["# Reproduction check", "",
             f"Outputs checked: {len(rows)}. " + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".",
             "", "**Result: " + ("FAILED" if failed else "PASSED") + "**", "",
             "| Output | Verdict | Detail |", "|---|---|---|"]
    lines += [f"| `{p}` | {v} | {d} |" for p, v, d in rows]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))
    for p, v, d in failed:
        print(f"  {v}: {p} ({d})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
