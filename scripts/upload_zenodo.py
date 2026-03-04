"""Upload staged archives to Zenodo via REST API.

Requires ZENODO_TOKEN environment variable (personal access token).

Usage:
    ZENODO_TOKEN=xxx uv run python scripts/upload_zenodo.py [--sandbox] [--dry-run] [--publish]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

STAGING_DIR = Path(__file__).resolve().parent.parent / "zenodo_staging"
ZENODO_JSON = Path(__file__).resolve().parent.parent / ".zenodo.json"

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"

# Archives that must be present for a valid upload
REQUIRED_ARCHIVES = [
    "benchmark_coexistence.tar.gz",
    "benchmark_single_family.tar.gz",
    "lenia_cross_substrate.tar.gz",
]

# (connect_timeout, read_timeout) in seconds
HTTP_TIMEOUT = (10, 300)


class UploadError(Exception):
    """Raised when upload preparation or execution fails."""


def _verify_checksums(staging: Path) -> bool:
    """Verify SHA256 checksums of archives against checksums.sha256."""
    checksum_file = staging / "checksums.sha256"
    if not checksum_file.exists():
        print("ERROR: checksums.sha256 not found. Re-run prepare_zenodo_metadata.py.")
        return False

    resolved_staging = staging.resolve()
    ok = True
    with open(checksum_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                print(f"  MALFORMED (line {line_num}): {line!r}")
                ok = False
                continue
            expected, name = parts

            # Path traversal protection
            path = (staging / name).resolve()
            try:
                path.relative_to(resolved_staging)
            except ValueError:
                print(f"  INVALID PATH: {name}")
                ok = False
                continue

            if not path.exists():
                print(f"  MISSING: {name}")
                ok = False
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected:
                print(f"  MISMATCH: {name} (expected {expected[:16]}..., got {actual[:16]}...)")
                ok = False
            else:
                print(f"  OK: {name}")
    return ok


def upload(*, sandbox: bool = False, dry_run: bool = False, publish: bool = False) -> None:
    """Upload staging directory to Zenodo.

    Raises:
        UploadError: If staging directory is invalid or checksums fail.
    """
    if not STAGING_DIR.exists():
        raise UploadError("Staging directory not found. Run prepare_zenodo_metadata.py first.")

    # Validate required archives are present
    missing = [name for name in REQUIRED_ARCHIVES if not (STAGING_DIR / name).exists()]
    if missing:
        raise UploadError(
            f"Required archives missing: {', '.join(missing)}. Re-run prepare_zenodo_metadata.py."
        )

    # Collect files to upload: archives + logs + checksums (skip .zenodo.json)
    upload_files = sorted(
        f for f in STAGING_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    if not upload_files:
        raise UploadError("No files found in staging directory.")

    # Verify checksums before upload
    print("Verifying checksums...")
    if not _verify_checksums(STAGING_DIR):
        raise UploadError("Checksum verification failed. Re-run prepare_zenodo_metadata.py.")

    print(f"\nFiles to upload ({len(upload_files)}):")
    total_size = 0
    for f in upload_files:
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name} ({size / 1e6:.1f} MB)")
    print(f"  Total: {total_size / 1e6:.1f} MB")

    if dry_run:
        print("\n[DRY RUN] Would create deposition and upload files. Exiting.")
        return

    try:
        import requests
    except ImportError as err:
        raise UploadError(
            "requests package required. Install with: uv sync --extra zenodo"
        ) from err

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise UploadError("ZENODO_TOKEN environment variable not set")

    api = SANDBOX_API if sandbox else ZENODO_API
    headers = {"Authorization": f"Bearer {token}"}

    # Load metadata
    with open(ZENODO_JSON) as f:
        metadata = json.load(f)

    # Create deposition
    print("\nCreating deposition...")
    r = requests.post(
        f"{api}/deposit/depositions",
        headers=headers,
        json={"metadata": metadata},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    deposition = r.json()
    dep_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"  Deposition ID: {dep_id}")

    # Upload files
    print(f"Uploading {len(upload_files)} files...")
    for filepath in upload_files:
        size_mb = filepath.stat().st_size / 1e6
        print(f"  {filepath.name} ({size_mb:.1f} MB)...", end=" ", flush=True)
        with open(filepath, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{filepath.name}",
                data=fp,
                headers={**headers, "Content-Type": "application/octet-stream"},
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
        print("done")

    # Optionally publish
    if publish:
        print("Publishing deposition...")
        r = requests.post(
            f"{api}/deposit/depositions/{dep_id}/actions/publish",
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        doi = r.json().get("doi", "N/A")
        print(f"  Published! DOI: {doi}")
    else:
        env_label = "sandbox." if sandbox else ""
        print(f"\nDeposition {dep_id} ready for review (DRAFT).")
        print(f"  URL: https://{env_label}zenodo.org/deposit/{dep_id}")
        print("  To publish: use --publish flag or log in to Zenodo and click 'Publish'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to Zenodo")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    parser.add_argument("--dry-run", action="store_true", help="Verify and list without uploading")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish after upload (default: draft)",
    )
    args = parser.parse_args()
    try:
        upload(sandbox=args.sandbox, dry_run=args.dry_run, publish=args.publish)
    except UploadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
