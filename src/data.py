"""Build monthly count series from the raw SPVM incident file, or from the frozen snapshot.

The analysis only uses each incident's month, category and district. The committed snapshot
of the download recorded in data/manifest.json holds exactly those counts, so when the raw
file is absent the incidents are rebuilt from it and every step runs unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "actes_criminels.csv"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
SNAPSHOTS = ROOT / "data" / "snapshots"

CATEGORY_LABELS = {
    "Introduction": "Break and enter",
    "Vol dans / sur véhicule à moteur": "Theft from vehicle",
    "Vol de véhicule à moteur": "Vehicle theft",
    "Méfait": "Mischief",
    "Vols qualifiés": "Robbery",
    "Infractions entrainant la mort": "Offences causing death",
}
# About 2-3 events per month citywide: too sparse for the count models used here.
EXCLUDED_FROM_FORECASTS = ["Offences causing death"]


def frozen_snapshot() -> Path:
    """The committed snapshot taken from the download recorded in data/manifest.json."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for meta_path in sorted(SNAPSHOTS.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("sha256") == manifest["sha256"]
                and meta.get("downloaded_utc") == manifest["downloaded_utc"]):
            return meta_path.with_suffix(".csv.gz")
    raise FileNotFoundError("No snapshot matches data/manifest.json")


def incidents_from_snapshot(path: Path) -> pd.DataFrame:
    """One row per incident, with DATE set to the first day of its month.

    Only the month, category and district are kept, which is all the analysis uses.
    """
    snap = pd.read_csv(path)
    rows = snap.loc[snap.index.repeat(snap["count"])]
    return pd.DataFrame({"CATEGORIE": rows["category"].to_numpy(),
                         "DATE": pd.to_datetime(rows["month"].to_numpy()),
                         "PDQ": rows["pdq"].to_numpy()})


def load_incidents(path: Path = RAW_PATH) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["DATE"])
    elif path == RAW_PATH:
        snapshot = frozen_snapshot()
        print(f"{path.name} not found: rebuilding incidents from {snapshot.relative_to(ROOT)}")
        df = incidents_from_snapshot(snapshot)
    else:
        raise FileNotFoundError(path)
    df["category"] = df["CATEGORIE"].map(CATEGORY_LABELS)
    if df["category"].isna().any():
        unknown = sorted(df.loc[df["category"].isna(), "CATEGORIE"].unique())
        raise ValueError(f"Unmapped categories: {unknown}")
    return df


def monthly_by_category(df: pd.DataFrame, drop_last_n_months: int = 1) -> pd.DataFrame:
    """Citywide monthly counts in long format (unique_id, ds, y).

    The final calendar month is partial at download time, so it is dropped.
    Recent months can still be revised by the SPVM; treat them with caution.
    """
    month = df["DATE"].dt.to_period("M").dt.to_timestamp()
    wide = df.groupby([month, "category"]).size().unstack(fill_value=0)
    wide = wide.asfreq("MS", fill_value=0)
    if drop_last_n_months:
        wide = wide.iloc[:-drop_last_n_months]
    wide = wide.drop(columns=EXCLUDED_FROM_FORECASTS)
    long = wide.stack().rename("y").reset_index()
    long.columns = ["ds", "unique_id", "y"]
    return long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


# Districts merged over the period, mapped to today's boundaries so each series is
# continuous. 22 -> 21 and 55 -> 5 are documented in the SPVM monthly report
# methodology; 24 -> 26 (2019), 11 -> 9 (2020) and 33 -> 31 (2021) in SPVM integration
# documents, consistent with where each absorbed district's counts end in this dataset.
PDQ_MERGERS = {11: 9, 24: 26, 33: 31, 22: 21, 55: 5}


def monthly_by_category_pdq(df: pd.DataFrame, drop_last_n_months: int = 1) -> pd.DataFrame:
    """Monthly counts per category and police district (wide: one column per pair).

    Columns are a MultiIndex (category, pdq). Rows without a district are dropped.
    """
    df = df.dropna(subset=["PDQ"]).copy()
    df["pdq"] = df["PDQ"].astype(int).replace(PDQ_MERGERS)
    df = df[~df["category"].isin(EXCLUDED_FROM_FORECASTS)]
    month = df["DATE"].dt.to_period("M").dt.to_timestamp()
    wide = df.groupby([month, "category", "pdq"]).size().unstack(["category", "pdq"], fill_value=0)
    wide = wide.asfreq("MS", fill_value=0).sort_index(axis=1)
    return wide.iloc[:-drop_last_n_months] if drop_last_n_months else wide
