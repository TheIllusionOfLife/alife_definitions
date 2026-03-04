"""Prepare Zenodo staging directory with tar.gz archives and checksums.

Creates three compressed archives from experiment data:
  - benchmark_coexistence.tar.gz  (benchmark/{E1-E5}/ + analysis artifacts)
  - benchmark_single_family.tar.gz (benchmark_single/{F1-F3}/{E1-E5}/)
  - lenia_cross_substrate.tar.gz  (lenia/*.json)

Also copies experiment logs and computes SHA256 checksums for all archives.

Usage:
    uv run python scripts/prepare_zenodo_metadata.py
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = REPO_ROOT / "zenodo_staging"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BENCHMARK_DIR = EXPERIMENTS_DIR / "benchmark"
BENCHMARK_SINGLE_DIR = EXPERIMENTS_DIR / "benchmark_single"
LENIA_DIR = EXPERIMENTS_DIR / "lenia"
ZENODO_JSON = REPO_ROOT / ".zenodo.json"

LOG_FILES = [
    "benchmark_log.txt",
    "benchmark_single_log.txt",
    "predictive_lineage_diversity.log",
    "surrogate_fpr_E1.log",
]

# Analysis artifacts that live in benchmark/ alongside the E* directories
ANALYSIS_ARTIFACTS = [
    "score_matrix.tsv",
    "agreement_analysis.json",
    "predictive_analysis.json",
    "frozen_thresholds.json",
    "benchmark_manifest.json",
    "loro.json",
    "sensitivity.json",
    "surrogate_fpr.json",
    "te_robustness.json",
    "temporal_d3.json",
    "bootstrap_ci.json",
]


def _is_regime_dir(name: str) -> bool:
    """Check if name matches E<digits> pattern (e.g. E1, E5)."""
    return len(name) >= 2 and name[0] == "E" and name[1:].isdigit()


def _is_family_dir(name: str) -> bool:
    """Check if name matches F<digits> pattern (e.g. F1, F3)."""
    return len(name) >= 2 and name[0] == "F" and name[1:].isdigit()


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_tar(archive_path: Path, base_name: str, source_dir: Path) -> int:
    """Create a gzip-compressed tar archive.

    Returns the number of files added.
    """
    count = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for f in sorted(source_dir.rglob("*")):
            if f.is_file():
                arcname = f"{base_name}/{f.relative_to(source_dir)}"
                tar.add(f, arcname=arcname)
                count += 1
    return count


def _archive_and_record(
    archive_path: Path,
    base_name: str,
    source_dir: Path,
    checksums: list[tuple[str, str]],
) -> None:
    """Create archive, compute checksum, append to checksums list, and print stats."""
    n_files = _make_tar(archive_path, base_name, source_dir)
    digest = _sha256(archive_path)
    checksums.append((digest, archive_path.name))
    size_mb = archive_path.stat().st_size / 1e6
    print(f"  -> {archive_path.name}: {n_files} files, {size_mb:.1f} MB, sha256={digest[:16]}...")


def prepare() -> None:
    """Prepare the staging directory with compressed archives."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    checksums: list[tuple[str, str]] = []

    # --- Archive 1: benchmark coexistence ---
    if BENCHMARK_DIR.exists():
        with tempfile.TemporaryDirectory(dir=STAGING_DIR) as tmp_str:
            tmp = Path(tmp_str)

            # Copy regime directories
            for regime_dir in sorted(BENCHMARK_DIR.iterdir()):
                if regime_dir.is_dir() and _is_regime_dir(regime_dir.name):
                    dest = tmp / regime_dir.name
                    dest.mkdir(parents=True)
                    for json_file in sorted(regime_dir.glob("seed_*.json")):
                        shutil.copy2(json_file, dest / json_file.name)
                    n = len(list(dest.glob("*.json")))
                    print(f"  {regime_dir.name}: {n} seed files")

            # Copy analysis artifacts
            for name in ANALYSIS_ARTIFACTS:
                src = BENCHMARK_DIR / name
                if src.exists():
                    shutil.copy2(src, tmp / name)
                    print(f"  Copied {name}")

            _archive_and_record(
                STAGING_DIR / "benchmark_coexistence.tar.gz",
                "benchmark_coexistence",
                tmp,
                checksums,
            )

    # --- Archive 2: benchmark single-family controls ---
    if BENCHMARK_SINGLE_DIR.exists():
        with tempfile.TemporaryDirectory(dir=STAGING_DIR) as tmp_str:
            tmp = Path(tmp_str)

            for family_dir in sorted(BENCHMARK_SINGLE_DIR.iterdir()):
                if not family_dir.is_dir() or not _is_family_dir(family_dir.name):
                    continue
                for regime_dir in sorted(family_dir.iterdir()):
                    if not regime_dir.is_dir() or not _is_regime_dir(regime_dir.name):
                        continue
                    dest = tmp / family_dir.name / regime_dir.name
                    dest.mkdir(parents=True)
                    for json_file in sorted(regime_dir.glob("seed_*.json")):
                        shutil.copy2(json_file, dest / json_file.name)
                    n = len(list(dest.glob("*.json")))
                    print(f"  {family_dir.name}/{regime_dir.name}: {n} seed files")

            # Copy manifest
            manifest = BENCHMARK_SINGLE_DIR / "benchmark_single_manifest.json"
            if manifest.exists():
                shutil.copy2(manifest, tmp / manifest.name)
                print(f"  Copied {manifest.name}")

            _archive_and_record(
                STAGING_DIR / "benchmark_single_family.tar.gz",
                "benchmark_single_family",
                tmp,
                checksums,
            )

    # --- Archive 3: Lenia cross-substrate (JSON only) ---
    if LENIA_DIR.exists():
        with tempfile.TemporaryDirectory(dir=STAGING_DIR) as tmp_str:
            tmp = Path(tmp_str)
            for json_file in sorted(LENIA_DIR.glob("*.json")):
                shutil.copy2(json_file, tmp / json_file.name)

            _archive_and_record(
                STAGING_DIR / "lenia_cross_substrate.tar.gz",
                "lenia_cross_substrate",
                tmp,
                checksums,
            )

    # --- Experiment logs ---
    for log_name in LOG_FILES:
        src = EXPERIMENTS_DIR / log_name
        if src.exists():
            shutil.copy2(src, STAGING_DIR / log_name)
            print(f"  Copied {log_name}")

    # --- .zenodo.json for reference ---
    if ZENODO_JSON.exists():
        shutil.copy2(ZENODO_JSON, STAGING_DIR / ".zenodo.json")

    # --- Write checksums ---
    checksum_path = STAGING_DIR / "checksums.sha256"
    with open(checksum_path, "w") as f:
        for digest, name in checksums:
            f.write(f"{digest}  {name}\n")
    print(f"\nChecksums written to {checksum_path.name}")

    # --- Summary ---
    total = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"\nStaging directory: {STAGING_DIR}")
    print(f"Total size: {total / 1e6:.1f} MB ({total / 1e9:.2f} GB)")
    print(f"Archives: {len(checksums)}")


if __name__ == "__main__":
    prepare()
