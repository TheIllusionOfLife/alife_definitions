"""Tests for the benchmark harness (scripts/experiment_benchmark.py).

Validates:
- Smoke test: 1 seed × 1 regime produces valid Mode B JSON with family_breakdown
- Resume: existing file is skipped when --resume is used
- Regime config: each E1–E5 produces a valid config dict
- Family count: Mode B output contains exactly 3 families per sample
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_module():
    """Import the benchmark module."""
    from scripts import experiment_benchmark

    return experiment_benchmark


@pytest.fixture(scope="module")
def smoke_result(benchmark_module, tmp_path_factory):
    """Run 1 seed × E1 baseline and return the parsed JSON result."""
    out_dir = tmp_path_factory.mktemp("benchmark_smoke")
    results = benchmark_module.run_benchmark(
        seeds=[0],
        regimes=["E1"],
        out_dir=out_dir,
        steps=200,
        sample_every=50,
    )
    assert len(results) == 1
    seed_key = ("E1", 0)
    assert seed_key in results
    return results[seed_key]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestBenchmarkSmoke:
    def test_result_has_samples(self, smoke_result):
        assert "samples" in smoke_result
        assert len(smoke_result["samples"]) > 0

    def test_result_has_family_breakdown(self, smoke_result):
        """Mode B runs must produce family_breakdown in each sample."""
        for sample in smoke_result["samples"]:
            assert "family_breakdown" in sample
            assert len(sample["family_breakdown"]) == 3

    def test_result_has_lineage_events(self, smoke_result):
        assert "lineage_events" in smoke_result

    def test_result_has_schema_version(self, smoke_result):
        assert smoke_result["schema_version"] == 1

    def test_json_written_to_disk(self, benchmark_module, tmp_path):
        """Running the benchmark writes per-seed JSON files."""
        benchmark_module.run_benchmark(
            seeds=[0],
            regimes=["E1"],
            out_dir=tmp_path,
            steps=200,
            sample_every=50,
        )
        json_path = tmp_path / "E1" / "seed_000.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "samples" in data


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestBenchmarkResume:
    def test_resume_skips_existing_file(self, benchmark_module, tmp_path):
        """With resume=True, existing seed files are skipped."""
        regime_dir = tmp_path / "E1"
        regime_dir.mkdir(parents=True)
        dummy = {"dummy": True, "samples": [], "schema_version": 1}
        seed_file = regime_dir / "seed_000.json"
        seed_file.write_text(json.dumps(dummy))

        benchmark_module.run_benchmark(
            seeds=[0],
            regimes=["E1"],
            out_dir=tmp_path,
            steps=200,
            sample_every=50,
            resume=True,
        )
        # The dummy file should be loaded, not overwritten
        reloaded = json.loads(seed_file.read_text())
        assert reloaded["dummy"] is True

    def test_resume_runs_missing_seeds(self, benchmark_module, tmp_path):
        """Resume runs seeds that don't have existing files."""
        results = benchmark_module.run_benchmark(
            seeds=[0],
            regimes=["E1"],
            out_dir=tmp_path,
            steps=200,
            sample_every=50,
            resume=True,
        )
        assert ("E1", 0) in results
        json_path = tmp_path / "E1" / "seed_000.json"
        assert json_path.exists()


# ---------------------------------------------------------------------------
# Regime configs
# ---------------------------------------------------------------------------


class TestRegimeConfigs:
    def test_all_regimes_produce_valid_config(self, benchmark_module):
        """Each regime name maps to a valid config override dict."""
        for regime in ["E1", "E2", "E3", "E4", "E5"]:
            config = benchmark_module.get_regime_overrides(regime)
            assert isinstance(config, dict)

    def test_e1_is_baseline(self, benchmark_module):
        config = benchmark_module.get_regime_overrides("E1")
        # E1 baseline has no overrides beyond families
        assert "resource_regeneration_rate" not in config
        assert "sensing_noise_scale" not in config

    def test_e2_sparse(self, benchmark_module):
        config = benchmark_module.get_regime_overrides("E2")
        assert config["resource_regeneration_rate"] == 0.005
        assert config["world_size"] == 150.0

    def test_e3_crowded(self, benchmark_module):
        config = benchmark_module.get_regime_overrides("E3")
        assert config["num_organisms"] == 80
        assert config["agents_per_organism"] == 30
        assert config["world_size"] == 80.0

    def test_e4_sensing_noise(self, benchmark_module):
        config = benchmark_module.get_regime_overrides("E4")
        assert config["sensing_noise_scale"] == 0.5

    def test_e5_spatial_patches(self, benchmark_module):
        config = benchmark_module.get_regime_overrides("E5")
        assert config["resource_patch_count"] == 4
        assert config["resource_patch_scale"] == 2.0

    def test_unknown_regime_raises(self, benchmark_module):
        with pytest.raises(ValueError, match="Unknown regime"):
            benchmark_module.get_regime_overrides("E99")


# ---------------------------------------------------------------------------
# Family count
# ---------------------------------------------------------------------------


class TestFamilyCount:
    def test_three_families_per_sample(self, smoke_result):
        """Mode B with 3 family profiles yields exactly 3 families."""
        for sample in smoke_result["samples"]:
            families = sample["family_breakdown"]
            family_ids = {f["family_id"] for f in families}
            assert family_ids == {0, 1, 2}
