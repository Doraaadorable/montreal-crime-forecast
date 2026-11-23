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
    same pixels       figures only: not byte-identical, but the decoded images are equal
    needs review      figures only: same size, and at most 5% of pixels differ. This does
                      not fail the check, because forecasts that move in the fifth
                      significant digit can shift a line by a fraction of a pixel, but a
                      small change can also be a real one (a legend entry, a point on a
                      curve), so these figures are listed for a person to look at, with the
                      share of changed pixels
    differs           a table with a larger difference, missing columns or rows; a figure
                      of another size, unreadable, or with more than 5% of pixels changed;
                      a page whose text differs, or whose numbers differ beyond tolerance
    missing           the rerun did not write the file

The check fails on "differs" or "missing", and flags "needs review" in the report, in the
run summary and as a warning on the run. Byte-identity of every file is reported but not
required, because floating-point results can vary slightly across machines.

Usage:
    python src/reproduce_check.py                # compare with HEAD, write the reports
"""
from __future__ import annotations

import hashlib
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from verify_outputs import tracked_outputs

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reproduce_report.md"
RTOL = 1e-4
ATOL = 1e-6
MAX_CHANGED_PIXELS = 0.05
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


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


def compare_images(new_bytes: bytes, old_bytes: bytes) -> tuple[str, str]:
    try:
        new = np.asarray(Image.open(io.BytesIO(new_bytes)).convert("RGBA"), dtype=np.int16)
        old = np.asarray(Image.open(io.BytesIO(old_bytes)).convert("RGBA"), dtype=np.int16)
    except OSError as exc:
        return "differs", f"cannot decode image ({exc})"
    if new.shape != old.shape:
        return "differs", f"size {old.shape[1]}x{old.shape[0]} -> {new.shape[1]}x{new.shape[0]}"
    changed = np.any(new != old, axis=2)
    share = float(changed.mean())
    if share == 0:
        return "same pixels", "decoded images are equal"
    detail = (f"{share:.3%} of pixels changed ({int(changed.sum())} of {changed.size}), "
              f"largest channel change {int(np.abs(new - old).max())} of 255")
    return ("needs review" if share <= MAX_CHANGED_PIXELS else "differs"), detail


def compare_page(new_text: str, old_text: str) -> tuple[str, str]:
    """Same text once numbers are masked, and numbers equal within tolerance."""
    if NUMBER.sub("#", new_text) != NUMBER.sub("#", old_text):
        return "differs", "text differs"
    new_nums, old_nums = NUMBER.findall(new_text), NUMBER.findall(old_text)
    worst = 0.0
    for a, b in zip(new_nums, old_nums):
        x, y = float(a), float(b)
        places = len(b.split(".")[1]) if "." in b else 0
        allowed = max(RTOL * abs(y) + ATOL, 10.0 ** -places * 1.0001)
        if abs(x - y) > allowed:
            return "differs", f"number {b} -> {a}"
        worst = max(worst, abs(x - y))
    return "within tolerance", f"text equal; largest number difference {worst:g}"


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
    if path.endswith(".png"):
        return compare_images(new_bytes, old_bytes)
    if path.endswith(".html"):
        return compare_page(new_bytes.decode("utf-8"), old_bytes.decode("utf-8"))
    return "differs", "no comparison defined for this file type"


def report(rows: list[tuple[str, str, str]]) -> tuple[list[str], int]:
    """Markdown report lines and the exit code for (path, verdict, detail) rows."""
    counts = pd.Series([r[1] for r in rows]).value_counts()
    failed = [r for r in rows if r[1] in ("differs", "missing")]
    review = [r for r in rows if r[1] == "needs review"]
    result = "FAILED" if failed else "PASSED"
    if review:
        result += f", {len(review)} figure{'s' if len(review) > 1 else ''} to review by eye"
    lines = ["# Reproduction check", "",
             f"Outputs checked: {len(rows)}. " + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".",
             "", f"**Result: {result}**", ""]
    if review:
        lines += ["Compare these with the committed versions in the run's artifact "
                  "(`figures/` against `committed/figures/`):", ""]
        lines += [f"- `{p}`: {d}" for p, _, d in review] + [""]
    lines += ["| Output | Verdict | Detail |", "|---|---|---|"]
    lines += [f"| `{p}` | {v} | {d} |" for p, v, d in rows]
    return lines, 1 if failed else 0


def main() -> int:
    rows = [(path, *check(path)) for path in tracked_outputs()]
    lines, code = report(rows)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))
    for p, v, d in rows:
        if v in ("differs", "missing"):
            print(f"  {v}: {p} ({d})")
        elif v == "needs review":
            print(f"::warning file={p}::Figure needs review: {d}")  # shown on the GitHub run
    return code


if __name__ == "__main__":
    sys.exit(main())
