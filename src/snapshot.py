"""Save a dated snapshot of monthly counts, to measure how published data is revised.

The SPVM can reclassify incidents after the first report, so counts for a given
month may change between downloads. Each snapshot stores counts by month, category
and district as they stood on the download date. Comparing snapshots shows how
much recent months move, which the main backtest cannot see.

Usage:
    python src/snapshot.py                 # snapshot data/raw/actes_criminels.csv
    python src/snapshot.py --compare       # revisions between first and latest snapshot
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "snapshots"
RAW = ROOT / "data" / "raw" / "actes_criminels.csv"
MANIFEST = ROOT / "data" / "manifest.json"
KEYS = ["month", "category", "pdq"]


def download_time(raw: Path, digest: str) -> tuple[datetime, str]:
    """When the raw file was downloaded, and where that time comes from.

    The manifest written by download_data.py records the download time and the file's hash.
    A file's modification time changes when it is copied, so it is used only when the
    manifest does not describe this file.
    """
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("sha256") == digest:
            stamp = datetime.strptime(manifest["downloaded_utc"], "%Y-%m-%dT%H:%M:%SZ")
            return stamp.replace(tzinfo=timezone.utc), "manifest"
    return datetime.fromtimestamp(raw.stat().st_mtime, tz=timezone.utc), "file modification time"


def take_snapshot(raw: Path = RAW) -> Path:
    df = pd.read_csv(raw, parse_dates=["DATE"])
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    downloaded, source = download_time(raw, digest)
    counts = (df.assign(month=df["DATE"].dt.strftime("%Y-%m"), pdq=df["PDQ"].astype("Int64"))
              .rename(columns={"CATEGORIE": "category"})
              .groupby(KEYS, dropna=False).size().rename("count").reset_index())
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    stem = f"{downloaded:%Y-%m-%d}"
    counts.to_csv(SNAPSHOTS / f"{stem}.csv.gz", index=False)
    meta = {"downloaded_utc": f"{downloaded:%Y-%m-%dT%H:%M:%SZ}",
            "downloaded_utc_source": source,
            "snapshot_created_utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
            "rows": len(df), "last_incident_date": f"{df['DATE'].max():%Y-%m-%d}",
            "sha256": digest}
    (SNAPSHOTS / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Snapshot {stem}: {len(df):,} incidents, {len(counts):,} cells")
    return SNAPSHOTS / f"{stem}.csv.gz"


def compare_snapshots() -> None:
    files = sorted(SNAPSHOTS.glob("*.csv.gz"))
    if len(files) < 2:
        print("Need at least two snapshots to measure revisions.")
        return
    first, latest = files[0], files[-1]
    a = pd.read_csv(first).groupby("month")["count"].sum()
    b = pd.read_csv(latest).groupby("month")["count"].sum()
    both = pd.DataFrame({"first": a, "latest": b}).dropna()
    both = both.iloc[:-1]  # the last month of the first snapshot was still in progress
    both["change"] = both["latest"] - both["first"]
    both["change_pct"] = (both["change"] / both["first"] * 100).round(2)
    print(f"Revisions from {first.name[:10]} to {latest.name[:10]}")
    print(f"Months changed: {(both['change'] != 0).sum()} of {len(both)}")
    print(both[both["change"] != 0].tail(24).to_string())
    both.to_csv(ROOT / "results" / "revisions.csv")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_snapshots()
    else:
        take_snapshot()
