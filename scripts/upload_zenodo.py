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


def _verify_checksums(staging: Path) -> bool:
    """Verify SHA256 checksums of archives against checksums.sha256."""
    checksum_file = staging / "checksums.sha256"
    if not checksum_file.exists():
        print("WARNING: checksums.sha256 not found, skipping verification")
        return True

    ok = True
    with open(checksum_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected, name = line.split("  ", 1)
            path = staging / name
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
    """Upload staging directory to Zenodo."""
    if not STAGING_DIR.exists():
        print("ERROR: Staging directory not found. Run prepare_zenodo_metadata.py first.")
        sys.exit(1)

    # Collect files to upload: archives + logs + checksums (skip .zenodo.json and temp dirs)
    upload_files = sorted(
        f for f in STAGING_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    if not upload_files:
        print("ERROR: No files found in staging directory.")
        sys.exit(1)

    # Verify checksums before upload
    print("Verifying checksums...")
    if not _verify_checksums(STAGING_DIR):
        print("ERROR: Checksum verification failed. Re-run prepare_zenodo_metadata.py.")
        sys.exit(1)

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
    except ImportError:
        print("ERROR: requests package required. Install with: uv sync --extra zenodo")
        sys.exit(1)

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN environment variable not set")
        sys.exit(1)

    api = SANDBOX_API if sandbox else ZENODO_API
    headers = {"Authorization": f"Bearer {token}"}

    # Load metadata
    with open(ZENODO_JSON) as f:
        metadata = json.load(f)

    # Create deposition
    print("\nCreating deposition...")
    r = requests.post(f"{api}/deposit/depositions", headers=headers, json={"metadata": metadata})
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
            )
            r.raise_for_status()
        print("done")

    # Optionally publish
    if publish:
        print("Publishing deposition...")
        r = requests.post(
            f"{api}/deposit/depositions/{dep_id}/actions/publish",
            headers=headers,
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
    upload(sandbox=args.sandbox, dry_run=args.dry_run, publish=args.publish)
