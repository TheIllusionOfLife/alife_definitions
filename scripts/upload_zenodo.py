"""Upload staged data to Zenodo via REST API.

Requires ZENODO_TOKEN environment variable (personal access token).
Use --sandbox flag for testing against sandbox.zenodo.org.

Usage:
    ZENODO_TOKEN=xxx uv run python scripts/upload_zenodo.py [--sandbox]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

STAGING_DIR = Path(__file__).resolve().parent.parent / "zenodo_staging"
ZENODO_JSON = Path(__file__).resolve().parent.parent / ".zenodo.json"

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"


def upload(sandbox: bool = False) -> None:
    """Upload staging directory to Zenodo."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests package required. Install with: uv pip install requests")
        sys.exit(1)

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN environment variable not set")
        sys.exit(1)

    api = SANDBOX_API if sandbox else ZENODO_API
    headers = {"Authorization": f"Bearer {token}"}

    if not STAGING_DIR.exists():
        print("ERROR: Staging directory not found. Run prepare_zenodo_metadata.py first.")
        sys.exit(1)

    # Load metadata
    with open(ZENODO_JSON) as f:
        metadata = json.load(f)

    # Create deposition
    print("Creating deposition...")
    r = requests.post(f"{api}/deposit/depositions", headers=headers, json={"metadata": metadata})
    r.raise_for_status()
    deposition = r.json()
    dep_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"  Deposition ID: {dep_id}")

    # Upload files
    files = sorted(STAGING_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"Uploading {len(files)} files...")

    for filepath in files:
        rel = filepath.relative_to(STAGING_DIR)
        # Zenodo uses flat filenames; encode path structure
        upload_name = str(rel).replace("/", "__")
        print(f"  {upload_name} ({filepath.stat().st_size / 1e6:.1f} MB)")
        with open(filepath, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{upload_name}",
                data=fp,
                headers={**headers, "Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()

    print(f"\nDeposition {dep_id} ready for review.")
    env_label = "sandbox." if sandbox else ""
    print(f"  URL: https://{env_label}zenodo.org/deposit/{dep_id}")
    print("  To publish: log in to Zenodo and click 'Publish'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to Zenodo")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    args = parser.parse_args()
    upload(sandbox=args.sandbox)
