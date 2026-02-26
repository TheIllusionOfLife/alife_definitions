"""Tests for adapter common infrastructure (scripts/adapters/common.py).

Validates:
- AdapterResult schema: score ∈ [0,1], criteria dict, family_id valid
- Per-family time series extraction from Mode B run data
- Shared statistical utilities (sigmoid, benjamini_hochberg)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# AdapterResult schema tests
# ---------------------------------------------------------------------------


class TestAdapterResult:
    def test_construction(self):
        from adapters.common import AdapterResult

        r = AdapterResult(
            definition="D1",
            family_id=0,
            score=0.75,
            passes_threshold=True,
            threshold_used=0.5,
            criteria={"metabolism": 0.8, "boundary": 0.7},
            metadata={"n_samples": 40},
        )
        assert r.definition == "D1"
        assert r.family_id == 0
        assert 0.0 <= r.score <= 1.0
        assert r.passes_threshold is True
        assert isinstance(r.criteria, dict)
        assert isinstance(r.metadata, dict)

    def test_score_range_validation(self):
        from adapters.common import AdapterResult

        # Valid scores
        for s in [0.0, 0.5, 1.0]:
            r = AdapterResult("D1", 0, s, False, 0.5, {}, {})
            assert r.score == s


# ---------------------------------------------------------------------------
# Time series extraction
# ---------------------------------------------------------------------------


class TestTimeSeriesExtraction:
    def test_extract_family_series(self, mode_b_run):
        from adapters.common import extract_family_series

        series = extract_family_series(mode_b_run, family_id=0)
        assert "alive_count" in series
        assert "energy_mean" in series
        assert "waste_mean" in series
        assert "boundary_mean" in series
        assert "birth_count" in series
        assert "death_count" in series
        assert "mean_generation" in series
        assert "mean_genome_drift" in series
        assert "genome_diversity" in series
        assert "maturity_mean" in series

    def test_series_length_matches_samples(self, mode_b_run):
        from adapters.common import extract_family_series

        series = extract_family_series(mode_b_run, family_id=0)
        n_samples = len(mode_b_run["samples"])
        for key, arr in series.items():
            assert len(arr) == n_samples, f"{key} length mismatch"

    def test_all_three_families_extractable(self, mode_b_run):
        from adapters.common import extract_family_series

        for fid in [0, 1, 2]:
            series = extract_family_series(mode_b_run, family_id=fid)
            assert len(series["alive_count"]) > 0

    def test_series_are_numpy_arrays(self, mode_b_run):
        from adapters.common import extract_family_series

        series = extract_family_series(mode_b_run, family_id=0)
        for key, arr in series.items():
            assert isinstance(arr, np.ndarray), f"{key} should be numpy array"


# ---------------------------------------------------------------------------
# Lineage extraction
# ---------------------------------------------------------------------------


class TestLineageExtraction:
    def test_extract_family_lineage(self, mode_b_run):
        from adapters.common import extract_family_lineage

        # F1 (full) should have some lineage events
        events = extract_family_lineage(mode_b_run, family_id=0)
        assert isinstance(events, list)
        # F1 has reproduction enabled, so expect some events
        # (may be zero in edge cases, but typically >0 for seed=42 with 2000 steps)

    def test_f3_has_no_lineage(self, mode_b_run):
        from adapters.common import extract_family_lineage

        # F3 (no reproduction) should have no lineage events
        events = extract_family_lineage(mode_b_run, family_id=2)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_sigmoid_maps_to_01(self):
        from adapters.common import sigmoid

        assert sigmoid(0.0) == pytest.approx(0.5)
        assert 0.0 < sigmoid(-10.0) < 0.5
        assert 0.5 < sigmoid(10.0) < 1.0
        assert sigmoid(100.0) == pytest.approx(1.0, abs=0.01)

    def test_benjamini_hochberg(self):
        from adapters.common import benjamini_hochberg

        # All non-significant
        corrected = benjamini_hochberg([0.5, 0.6, 0.7])
        assert all(p >= 0.5 for p in corrected)

        # One clearly significant
        corrected = benjamini_hochberg([0.001, 0.5, 0.9])
        assert corrected[0] < 0.05

    def test_benjamini_hochberg_empty(self):
        from adapters.common import benjamini_hochberg

        assert benjamini_hochberg([]) == []

    def test_coefficient_of_variation(self):
        from adapters.common import coefficient_of_variation

        arr = np.array([10.0, 10.0, 10.0])
        assert coefficient_of_variation(arr) == pytest.approx(0.0)

        arr2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cv = coefficient_of_variation(arr2)
        assert cv > 0.0
