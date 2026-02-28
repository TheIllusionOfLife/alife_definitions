"""Shared test fixtures for adapter and benchmark tests.

The mode_b_run fixture is session-scoped so all adapter test modules share
one ~30s simulation instead of each running its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path for adapter and experiment_common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import alife_defs
from experiment_common import FAMILY_PROFILES, TUNED_BASELINE


@pytest.fixture(scope="session")
def mode_b_run() -> dict:
    """Mode B run: 30 orgs, 25 agents, 2000 steps, sample_every=10, seed=42.

    Produces 200 sample points — sufficient for TE with 5 bins and
    lagged cross-correlation. Shared across all adapter test modules.
    """
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update(TUNED_BASELINE)
    cfg["seed"] = 42
    cfg["num_organisms"] = 30
    cfg["agents_per_organism"] = 25
    cfg["families"] = [dict(fp) for fp in FAMILY_PROFILES]
    result_json = alife_defs.run_experiment_json(json.dumps(cfg), 2000, 10)
    result = json.loads(result_json)
    result["regime_label"] = "E1"
    return result
