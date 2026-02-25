"""Binding contract tests: verify the PyO3 alife_defs module schema.

These tests use the real Rust binary (no mocks) to validate that the
JSON schema emitted by run_experiment_json() remains stable across refactors.
"""

from __future__ import annotations

import json

import alife_defs
import pytest

from scripts.experiment_common import run_single

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_OVERRIDE = {
    "num_organisms": 2,
    "agents_per_organism": 5,
    "world_size": 20.0,
    "seed": 42,
    "growth_maturation_steps": 10,
}


def _make_config(**overrides) -> str:
    """Build a minimal JSON config suitable for fast test runs."""
    base = json.loads(alife_defs.default_config_json())
    base.update(_MINIMAL_OVERRIDE)
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# default_config_json / validate_config_json round-trip
# ---------------------------------------------------------------------------


def test_default_config_round_trips_through_validate():
    """default_config_json() output must be accepted by validate_config_json()."""
    config_json = alife_defs.default_config_json()
    assert alife_defs.validate_config_json(config_json) is True


def test_default_config_is_valid_json():
    cfg = json.loads(alife_defs.default_config_json())
    assert isinstance(cfg, dict)
    assert cfg["num_organisms"] > 0
    assert cfg["world_size"] > 0


# ---------------------------------------------------------------------------
# run_experiment_json schema
# ---------------------------------------------------------------------------


def test_run_experiment_json_schema_version():
    """RunSummary must carry schema_version == 1."""
    result = json.loads(alife_defs.run_experiment_json(_make_config(), 10, 5))
    assert result["schema_version"] == 1


def test_run_experiment_json_required_top_level_fields():
    """All expected top-level keys must be present."""
    result = json.loads(alife_defs.run_experiment_json(_make_config(), 10, 5))
    required = {
        "schema_version",
        "steps",
        "sample_every",
        "final_alive_count",
        "samples",
        "lifespans",
        "total_reproduction_events",
        "lineage_events",
    }
    assert required.issubset(result.keys())


def test_run_experiment_json_types():
    """Top-level field types must match expected Python types."""
    result = json.loads(alife_defs.run_experiment_json(_make_config(), 10, 5))
    assert isinstance(result["steps"], int)
    assert isinstance(result["sample_every"], int)
    assert isinstance(result["final_alive_count"], int)
    assert isinstance(result["samples"], list)
    assert isinstance(result["lifespans"], list)
    assert isinstance(result["total_reproduction_events"], int)
    assert isinstance(result["lineage_events"], list)


def test_run_experiment_json_sample_fields():
    """Each StepMetrics sample must contain the expected metric fields."""
    result = json.loads(alife_defs.run_experiment_json(_make_config(), 10, 5))
    assert result["samples"], "Expected at least one sample"
    sample = result["samples"][0]
    required_sample_keys = {
        "step",
        "energy_mean",
        "waste_mean",
        "boundary_mean",
        "alive_count",
        "resource_total",
        "birth_count",
        "death_count",
        "population_size",
        "mean_generation",
        "mean_genome_drift",
        "agent_id_exhaustion_events",
        "energy_std",
        "waste_std",
        "boundary_std",
        "mean_age",
        "internal_state_mean",
        "internal_state_std",
        "genome_diversity",
        "max_generation",
        "maturity_mean",
        "spatial_cohesion_mean",
    }
    assert required_sample_keys.issubset(sample.keys()), (
        f"Missing keys: {required_sample_keys - sample.keys()}"
    )


def test_run_experiment_json_sample_count():
    """Number of samples must match ceil(steps / sample_every)."""
    steps, sample_every = 20, 5
    result = json.loads(alife_defs.run_experiment_json(_make_config(), steps, sample_every))
    expected = (steps + sample_every - 1) // sample_every
    assert len(result["samples"]) == expected


def test_run_experiment_json_steps_field_matches_argument():
    result = json.loads(alife_defs.run_experiment_json(_make_config(), 15, 5))
    assert result["steps"] == 15
    assert result["sample_every"] == 5


def test_validate_config_json_rejects_oversized_world():
    """validate_config_json() must reject world_size > MAX_WORLD_SIZE (2048.0)."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg["world_size"] = 99_999.0  # exceeds MAX_WORLD_SIZE — caught at config validation layer
    with pytest.raises(Exception, match="world_size"):
        alife_defs.validate_config_json(json.dumps(cfg))


# ---------------------------------------------------------------------------
# PR 1: regime_label + genome_hash additions
# ---------------------------------------------------------------------------


def test_regime_label_present():
    """regime_label must be stamped by the Python layer and round-trip correctly."""
    result = run_single(seed=0, overrides={**_MINIMAL_OVERRIDE}, regime_label="E1_baseline")
    assert result["regime_label"] == "E1_baseline"


def test_regime_label_defaults_to_empty_string():
    """regime_label defaults to empty string when not provided."""
    result = run_single(seed=0, overrides={**_MINIMAL_OVERRIDE})
    assert result["regime_label"] == ""


# ---------------------------------------------------------------------------
# PR 2a: FamilyConfig schema additions
# ---------------------------------------------------------------------------


def test_default_config_has_families_field():
    """families must be present in default config and be an empty list."""
    cfg = json.loads(alife_defs.default_config_json())
    assert "families" in cfg
    assert isinstance(cfg["families"], list)
    assert cfg["families"] == []


def test_families_field_roundtrips_through_validate():
    """A config with one FamilyConfig must pass validate_config_json."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg["families"] = [
        {
            "enable_reproduction": False,
            "initial_count": 5,
            "mutation_rate_multiplier": 0.5,
        }
    ]
    assert alife_defs.validate_config_json(json.dumps(cfg)) is True


def test_families_validation_rejects_zero_initial_count():
    cfg = json.loads(alife_defs.default_config_json())
    cfg["families"] = [{"initial_count": 0}]
    with pytest.raises(ValueError):
        alife_defs.validate_config_json(json.dumps(cfg))


# ---------------------------------------------------------------------------
# Family fixture constants (Mode B)
# ---------------------------------------------------------------------------

FAMILY_F1_FULL = {
    "enable_metabolism": True,
    "enable_boundary_maintenance": True,
    "enable_homeostasis": True,
    "enable_response": True,
    "enable_reproduction": True,
    "enable_evolution": True,
    "enable_growth": True,
    "initial_count": 10,
    "mutation_rate_multiplier": 1.0,
}

FAMILY_F2_DARWINIAN = {
    **FAMILY_F1_FULL,
    "enable_boundary_maintenance": False,
    "enable_homeostasis": False,
}

FAMILY_F3_AUTONOMY = {
    **FAMILY_F1_FULL,
    "enable_reproduction": False,
    "enable_evolution": False,
}


def test_lineage_event_has_genome_hash():
    """lineage_events items must carry a genome_hash int field.

    Uses overrides that make reproduction highly likely (low energy threshold,
    no boundary gate) to avoid a vacuous-pass when no events occur.
    """
    overrides = {
        **_MINIMAL_OVERRIDE,
        "reproduction_min_energy": 0.31,  # just above default cost (0.30)
        "reproduction_min_boundary": 0.0,  # remove boundary gate
    }
    result = run_single(seed=0, overrides=overrides, steps=500, sample_every=50)
    events = result["lineage_events"]
    assert events, "Expected at least one lineage event; genome_hash could not be verified."
    assert "genome_hash" in events[0], f"Missing genome_hash in lineage event: {events[0]}"
    assert isinstance(events[0]["genome_hash"], int)


# ---------------------------------------------------------------------------
# PR 2b: family_id on LineageEvent + Mode B smoke test
# ---------------------------------------------------------------------------


def test_lineage_event_has_family_id():
    """lineage_events items carry family_id int field in Mode B."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update({**_MINIMAL_OVERRIDE, "num_organisms": 4})
    cfg["families"] = [
        {**FAMILY_F1_FULL, "initial_count": 2},
        {**FAMILY_F1_FULL, "initial_count": 2},
    ]
    cfg["reproduction_min_energy"] = 0.31
    cfg["reproduction_min_boundary"] = 0.0
    result = json.loads(alife_defs.run_experiment_json(json.dumps(cfg), 500, 50))
    events = result["lineage_events"]
    assert events, "Expected at least one lineage event"
    for ev in events:
        assert "family_id" in ev, f"Missing family_id in lineage event: {ev}"
        assert isinstance(ev["family_id"], int), f"family_id is not int in: {ev}"


def test_mode_b_world_runs_without_error():
    """A 3-family Mode B config must complete a short run."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update({**_MINIMAL_OVERRIDE, "num_organisms": 30})
    cfg["families"] = [
        dict(FAMILY_F1_FULL),
        dict(FAMILY_F2_DARWINIAN),
        dict(FAMILY_F3_AUTONOMY),
    ]
    result = json.loads(alife_defs.run_experiment_json(json.dumps(cfg), 20, 5))
    assert result["schema_version"] == 1
