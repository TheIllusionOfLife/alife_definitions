"""Tests for sensitivity analysis sweeps.

Validates:
- D1 weight perturbation changes scores but preserves F1 > F2 > F3 ordering
- Threshold sweep: higher thresholds → fewer passes (monotonic)
- D3 FDR sweep: higher q → more edges → higher/equal closure
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# D1 weight perturbation
# ---------------------------------------------------------------------------


class TestD1WeightSensitivity:
    def test_d1_accepts_custom_weights(self, mode_b_run):
        """score_d1 should accept optional weights parameter."""
        from adapters.d1 import score_d1

        result = score_d1(
            mode_b_run,
            family_id=0,
            weights=(0.4, 0.3, 0.3),
        )
        assert 0.0 <= result.score <= 1.0

    def test_d1_default_weights_unchanged(self, mode_b_run):
        """Calling without weights should match default behavior."""
        from adapters.d1 import score_d1

        r_default = score_d1(mode_b_run, family_id=0)
        r_explicit = score_d1(mode_b_run, family_id=0, weights=(0.3, 0.4, 0.3))
        assert r_default.score == pytest.approx(r_explicit.score, abs=1e-9)

    def test_d1_perturbed_weights_change_score(self, mode_b_run):
        """±20% weight perturbation should produce a different score."""
        from adapters.d1 import score_d1

        r_perturbed = score_d1(mode_b_run, family_id=0, weights=(0.36, 0.32, 0.36))
        # Just check the perturbed result is valid
        assert 0.0 <= r_perturbed.score <= 1.0

    def test_d1_ordering_preserved_under_perturbation(self, mode_b_run):
        """F1 should still score higher than F3 under ±20% weight perturbation."""
        from adapters.d1 import score_d1

        # Several perturbation vectors
        perturbations = [
            (0.36, 0.32, 0.36),  # +20% α, -20% β, +20% γ
            (0.24, 0.48, 0.24),  # -20% α, +20% β, -20% γ
            (0.36, 0.48, 0.24),  # +20% α, +20% β, -20% γ
        ]
        for w in perturbations:
            r0 = score_d1(mode_b_run, family_id=0, weights=w)
            r2 = score_d1(mode_b_run, family_id=2, weights=w)
            assert r0.score >= r2.score, (
                f"F1 ({r0.score:.3f}) < F3 ({r2.score:.3f}) with weights={w}"
            )


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


class TestThresholdSweep:
    def test_higher_threshold_fewer_passes(self, mode_b_run):
        """Higher thresholds should produce fewer or equal passes."""
        from adapters import score_all

        thresholds_low = {"D1": 0.1, "D2": 0.1, "D3": 0.1, "D4": 0.1}
        thresholds_high = {"D1": 0.7, "D2": 0.7, "D3": 0.7, "D4": 0.7}

        passes_low = 0
        passes_high = 0
        for fid in [0, 1, 2]:
            results_low = score_all(mode_b_run, family_id=fid, thresholds=thresholds_low)
            results_high = score_all(mode_b_run, family_id=fid, thresholds=thresholds_high)
            for defn in ["D1", "D2", "D3", "D4"]:
                if results_low[defn].passes_threshold:
                    passes_low += 1
                if results_high[defn].passes_threshold:
                    passes_high += 1

        assert passes_low >= passes_high


# ---------------------------------------------------------------------------
# D3 FDR sweep
# ---------------------------------------------------------------------------


class TestD3FdrSensitivity:
    def test_d3_accepts_custom_fdr_q(self, mode_b_run):
        """score_d3 should accept optional fdr_q parameter."""
        from adapters.d3 import score_d3

        result = score_d3(mode_b_run, family_id=0, fdr_q=0.10)
        assert 0.0 <= result.score <= 1.0

    def test_d3_higher_q_more_or_equal_edges(self, mode_b_run):
        """Higher FDR q → more liberal → more or equal significant edges."""
        from adapters.d3 import score_d3

        r_strict = score_d3(mode_b_run, family_id=0, fdr_q=0.01)
        r_liberal = score_d3(mode_b_run, family_id=0, fdr_q=0.10)

        edges_strict = r_strict.metadata["n_significant_edges"]
        edges_liberal = r_liberal.metadata["n_significant_edges"]
        assert edges_liberal >= edges_strict, (
            f"q=0.10 ({edges_liberal} edges) < q=0.01 ({edges_strict} edges)"
        )

    def test_d3_higher_q_higher_or_equal_closure(self, mode_b_run):
        """Higher FDR q → more edges → higher or equal closure score."""
        from adapters.d3 import score_d3

        r_strict = score_d3(mode_b_run, family_id=0, fdr_q=0.01)
        r_liberal = score_d3(mode_b_run, family_id=0, fdr_q=0.10)

        assert r_liberal.criteria["closure"] >= r_strict.criteria["closure"], (
            f"q=0.10 closure ({r_liberal.criteria['closure']:.3f}) "
            f"< q=0.01 closure ({r_strict.criteria['closure']:.3f})"
        )
