"""Build monthly count series from the raw SPVM incident file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "actes_criminels.csv"

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


def load_incidents(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])
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
