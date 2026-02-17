"""Helpers for experiment run manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sorted_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def config_digest(config: dict) -> str:
    return hashlib.sha256(_sorted_json(config).encode("utf-8")).hexdigest()


def write_manifest(
    out_path: Path,
    *,
    experiment_name: str,
    steps: int,
    sample_every: int,
    seeds: list[int],
    base_config: dict,
    condition_overrides: dict[str, dict],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "steps": steps,
        "sample_every": sample_every,
        "seeds": seeds,
        "base_config": base_config,
        "base_config_digest": config_digest(base_config),
        "condition_overrides": condition_overrides,
        "condition_config_digests": {
            name: config_digest({**base_config, **overrides})
            for name, overrides in condition_overrides.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
