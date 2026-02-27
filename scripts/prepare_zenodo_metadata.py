"""Prepare Zenodo metadata and staging directory for data upload.

Copies benchmark data, score matrix, and analysis results into a staging
directory with the correct structure for Zenodo archival.

Usage:
    uv run python scripts/prepare_zenodo_metadata.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = REPO_ROOT / "zenodo_staging"
BENCHMARK_DIR = REPO_ROOT / "experiments" / "benchmark"
ZENODO_JSON = REPO_ROOT / ".zenodo.json"


def prepare() -> None:
    """Prepare the staging directory."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    # Copy benchmark data
    if BENCHMARK_DIR.exists():
        for regime_dir in sorted(BENCHMARK_DIR.iterdir()):
            if regime_dir.is_dir() and regime_dir.name.startswith("E"):
                dest = STAGING_DIR / "benchmark" / regime_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for json_file in sorted(regime_dir.glob("seed_*.json")):
                    shutil.copy2(json_file, dest / json_file.name)
                n = len(list(dest.glob("*.json")))
                print(f"  {regime_dir.name}: {n} seed files")

    # Copy analysis artifacts
    for name in ["score_matrix.tsv", "agreement_analysis.json",
                 "predictive_analysis.json", "frozen_thresholds.json",
                 "benchmark_manifest.json"]:
        src = BENCHMARK_DIR / name
        if src.exists():
            shutil.copy2(src, STAGING_DIR / name)
            print(f"  Copied {name}")

    # Copy .zenodo.json for reference
    if ZENODO_JSON.exists():
        shutil.copy2(ZENODO_JSON, STAGING_DIR / ".zenodo.json")

    # Compute total size
    total = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"\nStaging directory: {STAGING_DIR}")
    print(f"Total size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    prepare()
