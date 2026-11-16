"""Check that the committed outputs match what the code produces.

    python src/verify_outputs.py           # compare outputs with the recorded hashes
    python src/verify_outputs.py --write   # record hashes and the software environment

Byte-identical output is only expected with the same data file (data/manifest.json)
and the package versions in requirements.txt. Other versions can change the last
digits of a number or the encoding of an image without changing any conclusion.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = ROOT / "results" / "output_hashes.sha256"
ENVIRONMENT = ROOT / "results" / "environment.txt"
PACKAGES = ["pandas", "numpy", "statsforecast", "pyarrow", "matplotlib", "lightgbm",
            "scikit-learn", "scipy"]


def tracked_outputs() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "results", "figures", "docs/index.html"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    skip = {HASHES.relative_to(ROOT).as_posix(), ENVIRONMENT.relative_to(ROOT).as_posix()}
    return sorted(f for f in listed if f not in skip)


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write() -> None:
    files = tracked_outputs()
    HASHES.write_text("".join(f"{digest(f)}  {f}\n" for f in files), encoding="utf-8")
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
             f"cpu_count {os.cpu_count()}"] + [f"{p}=={version(p)}" for p in PACKAGES]
    ENVIRONMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded {len(files)} output hashes and the environment")


def verify() -> int:
    recorded = dict(reversed(line.split("  ", 1)) for line in HASHES.read_text().splitlines())
    changed = [f for f, h in recorded.items() if not (ROOT / f).exists() or digest(f) != h]
    print(f"{len(recorded) - len(changed)} of {len(recorded)} outputs match")
    for f in changed:
        print("  differs:", f)
    return 1 if changed else 0


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
    else:
        sys.exit(verify())
