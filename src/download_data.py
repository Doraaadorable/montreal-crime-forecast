"""Download the SPVM "Actes criminels" incident dataset via the CKAN datastore API.

Source: Ville de Montréal open data portal (CC BY 4.0)
https://donnees.montreal.ca/dataset/actes-criminels

Usage:
    python src/download_data.py            # writes data/raw/actes_criminels.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://donnees.montreal.ca/api/3/action/datastore_search"
RESOURCE_ID = "c6f482bf-bf0f-4960-8b2f-9982c211addd"
FIELDS = ["_id", "CATEGORIE", "DATE", "QUART", "PDQ", "LONGITUDE", "LATITUDE"]
PAGE_SIZE = 32000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "actes_criminels.csv"


def fetch_page(offset: int, retries: int = 4) -> dict:
    query = urllib.parse.urlencode(
        {
            "resource_id": RESOURCE_ID,
            "limit": PAGE_SIZE,
            "offset": offset,
            "fields": ",".join(FIELDS),
            "sort": "_id",
        }
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=120) as resp:
                payload = json.load(resp)
            if not payload.get("success"):
                raise RuntimeError(f"API returned success=false at offset {offset}")
            return payload["result"]
        except Exception as exc:  # network hiccups: back off and retry
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s ({exc})", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    offset, total, written = 0, None, 0
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        while total is None or offset < total:
            result = fetch_page(offset)
            total = result["total"]
            records = result["records"]
            if not records:
                break
            writer.writerows({k: r.get(k) for k in FIELDS} for r in records)
            written += len(records)
            offset += len(records)
            print(f"  {written:,} / {total:,}")
    if written != total:
        raise RuntimeError(f"Expected {total} rows, wrote {written}")
    print(f"Saved {written:,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
