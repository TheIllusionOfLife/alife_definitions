"""Tests for single-family benchmark harness.

Validates:
- Single-family config has exactly 1 family entry
- Short run completes and produces family_breakdown
- Adapters score single-family runs without error
"""

from __future__ import annotations

import json

import alife_defs
import pytest
from experiment_common import FAMILY_PROFILES


class TestSingleFamilyConfig:
    def test_config_has_one_family(self):
        from experiment_benchmark_single import _build_single_family_config

        config = _build_single_family_config(
            seed=0,
            family_profile=dict(FAMILY_PROFILES[0]),
            regime_overrides={},
        )
        assert len(config["families"]) == 1
        assert config["num_organisms"] == FAMILY_PROFILES[0]["initial_count"]

    def test_config_applies_regime_overrides(self):
        from experiment_benchmark_single import _build_single_family_config

        config = _build_single_family_config(
            seed=0,
            family_profile=dict(FAMILY_PROFILES[0]),
            regime_overrides={"sensing_noise_scale": 0.5},
        )
        assert config["sensing_noise_scale"] == 0.5


class TestSingleFamilySmokeRun:
    @pytest.fixture(scope="class")
    def single_family_run(self):
        """Run a minimal single-family experiment."""
        from experiment_benchmark_single import _build_single_family_config

        config = _build_single_family_config(
            seed=42,
            family_profile=dict(FAMILY_PROFILES[0]),
            regime_overrides={},
        )
        # Override for fast test
        config["num_organisms"] = 5
        config["agents_per_organism"] = 5
        config["families"][0]["initial_count"] = 5
        result_json = alife_defs.run_experiment_json(json.dumps(config), 100, 10)
        return json.loads(result_json)

    def test_run_completes(self, single_family_run):
        assert single_family_run["schema_version"] == 1
        assert len(single_family_run["samples"]) > 0

    def test_has_family_breakdown(self, single_family_run):
        """Single-family Mode B should produce family_breakdown with 1 entry."""
        for sample in single_family_run["samples"]:
            breakdown = sample.get("family_breakdown", [])
            assert len(breakdown) == 1, f"Expected 1 family_breakdown entry, got {len(breakdown)}"
            assert breakdown[0]["family_id"] == 0

    def test_adapters_score_without_error(self, single_family_run):
        """All four adapters should score a single-family run without error."""
        from adapters import score_all

        results = score_all(single_family_run, family_id=0)
        assert len(results) == 4
        for defn, result in results.items():
            assert 0.0 <= result.score <= 1.0, f"{defn} score out of range: {result.score}"
