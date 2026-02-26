"""Tests for D1 adapter — Textbook 7-criteria + Functional Analogy.

Validates:
- F1 (all 7 criteria) scores strictly higher than F2 and F3
- Each sub-criterion returns a score in [0,1]
- Ablation contrasts: F2 lacks boundary/homeostasis → those scores < F1's
- Coupling sub-scores: at least some TE edges significant for F1
- Geometric mean aggregation: zero sub-criterion pulls total toward zero
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import alife_defs
from experiment_common import FAMILY_PROFILES, TUNED_BASELINE

# ---------------------------------------------------------------------------
# Shared fixture — reuse the same Mode B run for all D1 tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mode_b_run() -> dict:
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update(TUNED_BASELINE)
    cfg["seed"] = 42
    cfg["num_organisms"] = 30
    cfg["agents_per_organism"] = 25
    cfg["families"] = [dict(fp) for fp in FAMILY_PROFILES]
    result_json = alife_defs.run_experiment_json(json.dumps(cfg), 2000, 50)
    result = json.loads(result_json)
    result["regime_label"] = "E1"
    return result


@pytest.fixture(scope="module")
def d1_scores(mode_b_run):
    """Score all 3 families with D1."""
    from adapters.d1 import score_d1

    return {fid: score_d1(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


# ---------------------------------------------------------------------------
# Score range tests
# ---------------------------------------------------------------------------


class TestD1ScoreRange:
    def test_all_scores_in_unit_interval(self, d1_scores):
        for fid, result in d1_scores.items():
            assert 0.0 <= result.score <= 1.0, f"F{fid + 1} score out of range: {result.score}"

    def test_criteria_dict_has_seven_keys(self, d1_scores):
        expected = {
            "metabolism",
            "boundary",
            "homeostasis",
            "response",
            "reproduction",
            "evolution",
            "growth",
        }
        for fid, result in d1_scores.items():
            assert set(result.criteria.keys()) == expected, (
                f"F{fid + 1} criteria keys mismatch: {set(result.criteria.keys())}"
            )

    def test_each_criterion_in_unit_interval(self, d1_scores):
        for fid, result in d1_scores.items():
            for name, val in result.criteria.items():
                assert 0.0 <= val <= 1.0, f"F{fid + 1} criterion '{name}' out of range: {val}"


# ---------------------------------------------------------------------------
# Cross-family ordering
# ---------------------------------------------------------------------------


class TestD1FamilyOrdering:
    def test_f1_scores_higher_than_f2(self, d1_scores):
        """F1 (full) should score higher than F2 (no boundary/homeostasis)."""
        assert d1_scores[0].score > d1_scores[1].score

    def test_f1_scores_higher_than_f3(self, d1_scores):
        """F1 (full) should score higher than F3 (no reproduction/evolution)."""
        assert d1_scores[0].score > d1_scores[2].score


# ---------------------------------------------------------------------------
# Ablation contrasts
# ---------------------------------------------------------------------------


class TestD1AblationContrasts:
    def test_f2_boundary_lower_than_f1(self, d1_scores):
        """F2 lacks boundary → boundary criterion should score lower than F1."""
        assert d1_scores[1].criteria["boundary"] < d1_scores[0].criteria["boundary"]

    def test_f2_homeostasis_lower_than_f1(self, d1_scores):
        """F2 lacks homeostasis → homeostasis criterion should score lower than F1."""
        assert d1_scores[1].criteria["homeostasis"] < d1_scores[0].criteria["homeostasis"]

    def test_f3_reproduction_lower_than_f1(self, d1_scores):
        """F3 lacks reproduction → reproduction criterion should score lower."""
        assert d1_scores[2].criteria["reproduction"] < d1_scores[0].criteria["reproduction"]

    def test_f3_evolution_lower_than_f1(self, d1_scores):
        """F3 lacks evolution → evolution criterion should score lower."""
        assert d1_scores[2].criteria["evolution"] < d1_scores[0].criteria["evolution"]


# ---------------------------------------------------------------------------
# Coupling sub-scores
# ---------------------------------------------------------------------------


class TestD1Coupling:
    def test_f1_has_some_coupling(self, d1_scores):
        """F1 should have at least some non-zero coupling evidence."""
        meta = d1_scores[0].metadata
        coupling_scores = meta.get("coupling_scores", {})
        # At least one criterion should have non-zero coupling
        assert any(v > 0 for v in coupling_scores.values()), (
            f"F1 coupling scores all zero: {coupling_scores}"
        )


# ---------------------------------------------------------------------------
# Geometric mean property
# ---------------------------------------------------------------------------


class TestD1GeometricMean:
    def test_geometric_mean_of_zeros_is_zero(self):
        """If any criterion is zero, geometric mean should be zero."""
        from adapters.d1 import _geometric_mean

        vals = [0.8, 0.0, 0.9, 0.7, 0.6, 0.5, 0.4]
        assert _geometric_mean(vals) == pytest.approx(0.0)

    def test_geometric_mean_of_ones(self):
        from adapters.d1 import _geometric_mean

        vals = [1.0] * 7
        assert _geometric_mean(vals) == pytest.approx(1.0)

    def test_geometric_mean_less_than_arithmetic(self):
        from adapters.d1 import _geometric_mean

        vals = [0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8]
        gm = _geometric_mean(vals)
        am = sum(vals) / len(vals)
        assert gm <= am


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


class TestD1Metadata:
    def test_definition_label(self, d1_scores):
        for _fid, result in d1_scores.items():
            assert result.definition == "D1"

    def test_family_id_matches(self, d1_scores):
        for fid, result in d1_scores.items():
            assert result.family_id == fid
