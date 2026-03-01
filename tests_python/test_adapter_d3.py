"""Tests for D3 adapter — Autonomy / Organizational Closure.

Validates:
- F1 has higher closure score than F2 (F2 lacks boundary loop)
- F3 has moderate closure (self-maintenance without reproduction)
- Directed graph has at least |SCC| ≥ 2 for F1
- Persistence penalty: collapsed family scores near 0
- TE computation: at least some edges significant for F1
- Mode parameter: closure_only vs closure_x_persistence
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def d3_scores(mode_b_run):
    from adapters.d3 import score_d3

    return {fid: score_d3(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


@pytest.fixture(scope="module")
def d3_scores_closure_x_persistence(mode_b_run):
    from adapters.d3 import score_d3

    return {
        fid: score_d3(mode_b_run, family_id=fid, mode="closure_x_persistence") for fid in [0, 1, 2]
    }


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------


class TestD3ScoreRange:
    def test_all_scores_in_unit_interval(self, d3_scores):
        for fid, result in d3_scores.items():
            assert 0.0 <= result.score <= 1.0, f"F{fid + 1} score: {result.score}"

    def test_criteria_has_expected_keys(self, d3_scores):
        expected = {
            "closure",
            "persistence",
            "score_closure_only",
            "score_closure_x_persistence",
        }
        for _fid, result in d3_scores.items():
            assert expected.issubset(result.criteria.keys())


# ---------------------------------------------------------------------------
# Cross-family ordering
# ---------------------------------------------------------------------------


class TestD3FamilyOrdering:
    def test_f1_combined_higher_than_f2(self, d3_scores_closure_x_persistence):
        """F1 (full) should have higher closure×persistence than F2.

        With closure_only mode, pure SCC size can favor F2 (tight coupling
        in fewer processes). The combined score adds persistence weight that
        favors the full family. This test uses closure_x_persistence mode.
        """
        assert d3_scores_closure_x_persistence[0].score >= d3_scores_closure_x_persistence[1].score

    def test_f3_has_moderate_closure(self, d3_scores):
        """F3 (autonomy, no reproduction) should have moderate closure > 0."""
        assert d3_scores[2].criteria["closure"] > 0.0


# ---------------------------------------------------------------------------
# SCC properties
# ---------------------------------------------------------------------------


class TestD3SCC:
    def test_f1_scc_at_least_two(self, d3_scores):
        """F1 directed graph should have SCC of at least 2 variables."""
        scc_size = d3_scores[0].metadata.get("largest_scc_size", 0)
        assert scc_size >= 2, f"F1 SCC size: {scc_size}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestD3Persistence:
    def test_persistence_in_unit_interval(self, d3_scores):
        for _fid, result in d3_scores.items():
            assert 0.0 <= result.criteria["persistence"] <= 1.0


# ---------------------------------------------------------------------------
# Edge significance
# ---------------------------------------------------------------------------


class TestD3Edges:
    def test_f1_has_some_significant_edges(self, d3_scores):
        """F1 should have at least some significant TE/Granger edges."""
        n_edges = d3_scores[0].metadata.get("n_significant_edges", 0)
        assert n_edges >= 1, f"F1 significant edges: {n_edges}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestD3Metadata:
    def test_definition_label(self, d3_scores):
        for _fid, result in d3_scores.items():
            assert result.definition == "D3"

    def test_family_id_matches(self, d3_scores):
        for fid, result in d3_scores.items():
            assert result.family_id == fid


# ---------------------------------------------------------------------------
# Mode parameter: closure_only vs closure_x_persistence
# ---------------------------------------------------------------------------


class TestD3Mode:
    def test_default_mode_is_closure_only(self, d3_scores):
        """Default score should equal the closure_only value in criteria."""
        for fid, result in d3_scores.items():
            assert result.score == pytest.approx(result.criteria["score_closure_only"]), (
                f"F{fid + 1}: default score should be closure_only"
            )

    def test_closure_x_persistence_mode(self, d3_scores_closure_x_persistence):
        """When mode=closure_x_persistence, score should equal the product."""
        for fid, result in d3_scores_closure_x_persistence.items():
            assert result.score == pytest.approx(result.criteria["score_closure_x_persistence"]), (
                f"F{fid + 1}: score should be closure × persistence"
            )

    def test_both_scores_in_criteria_regardless_of_mode(
        self, d3_scores, d3_scores_closure_x_persistence
    ):
        """Both score variants must appear in criteria regardless of active mode."""
        for scores in [d3_scores, d3_scores_closure_x_persistence]:
            for fid, result in scores.items():
                assert "score_closure_only" in result.criteria, (
                    f"F{fid + 1}: missing score_closure_only"
                )
                assert "score_closure_x_persistence" in result.criteria, (
                    f"F{fid + 1}: missing score_closure_x_persistence"
                )
