"""Prepare Zenodo metadata and staging directory for data upload.

Copies benchmark data, score matrix, and analysis results into a staging
directory with the correct structure for Zenodo archival.

Usage:
    uv run python scripts/prepare_zenodo_metadata.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = REPO_ROOT / "zenodo_staging"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BENCHMARK_DIR = EXPERIMENTS_DIR / "benchmark"
BENCHMARK_SINGLE_DIR = EXPERIMENTS_DIR / "benchmark_single"
ZENODO_JSON = REPO_ROOT / ".zenodo.json"

LOG_FILES = [
    "benchmark_log.txt",
    "benchmark_single_log.txt",
    "predictive_lineage_diversity.log",
    "surrogate_fpr_E1.log",
]


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

    # Copy benchmark_single data (single-family controls)
    if BENCHMARK_SINGLE_DIR.exists():
        for family_dir in sorted(BENCHMARK_SINGLE_DIR.iterdir()):
            if not family_dir.is_dir() or not family_dir.name.startswith("F"):
                continue
            for regime_dir in sorted(family_dir.iterdir()):
                if not regime_dir.is_dir() or not regime_dir.name.startswith("E"):
                    continue
                dest = STAGING_DIR / "benchmark_single" / family_dir.name / regime_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for json_file in sorted(regime_dir.glob("seed_*.json")):
                    shutil.copy2(json_file, dest / json_file.name)
                n = len(list(dest.glob("*.json")))
                print(f"  {family_dir.name}/{regime_dir.name}: {n} seed files")
        # Copy single-family manifest if present
        manifest = BENCHMARK_SINGLE_DIR / "benchmark_single_manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, STAGING_DIR / "benchmark_single" / manifest.name)
            print(f"  Copied {manifest.name}")

    # Copy experiment logs
    logs_dest = STAGING_DIR / "logs"
    logs_dest.mkdir(parents=True, exist_ok=True)
    for log_name in LOG_FILES:
        src = EXPERIMENTS_DIR / log_name
        if src.exists():
            shutil.copy2(src, logs_dest / log_name)
            print(f"  Copied logs/{log_name}")

    # Copy analysis artifacts
    for name in [
        "score_matrix.tsv",
        "agreement_analysis.json",
        "predictive_analysis.json",
        "frozen_thresholds.json",
        "benchmark_manifest.json",
    ]:
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
