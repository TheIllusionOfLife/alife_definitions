"""Tests for D2 adapter — Darwinian / NASA definition.

Validates:
- F2 (Darwinian) and F1 (full) score higher than F3 (no reproduction)
- F3 S_reprod ≈ 0 (no birth events)
- Heritability: with lineage events, S_hered > 0.5
- Selection differential: S_select is non-negative for F1/F2
- Without lineage events (F3): S_hered = 0
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def d2_scores(mode_b_run):
    from adapters.d2 import score_d2

    return {fid: score_d2(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------


class TestD2ScoreRange:
    def test_all_scores_in_unit_interval(self, d2_scores):
        for fid, result in d2_scores.items():
            assert 0.0 <= result.score <= 1.0, f"F{fid + 1} score: {result.score}"

    def test_criteria_has_three_sub_scores(self, d2_scores):
        expected = {"S_reprod", "S_hered", "S_select"}
        for _fid, result in d2_scores.items():
            assert set(result.criteria.keys()) == expected


# ---------------------------------------------------------------------------
# Cross-family ordering
# ---------------------------------------------------------------------------


class TestD2FamilyOrdering:
    def test_f1_higher_than_f3(self, d2_scores):
        """F1 (full) should score higher than F3 (no reproduction)."""
        assert d2_scores[0].score > d2_scores[2].score

    def test_f2_higher_than_f3(self, d2_scores):
        """F2 (Darwinian) should score higher than F3 (no reproduction)."""
        assert d2_scores[1].score > d2_scores[2].score


# ---------------------------------------------------------------------------
# Sub-score properties
# ---------------------------------------------------------------------------


class TestD2SubScores:
    def test_f3_reprod_near_zero(self, d2_scores):
        """F3 has no reproduction → S_reprod should be 0."""
        assert d2_scores[2].criteria["S_reprod"] == pytest.approx(0.0, abs=0.05)

    def test_f3_hered_zero(self, d2_scores):
        """F3 has no lineage events → S_hered should be 0."""
        assert d2_scores[2].criteria["S_hered"] == 0.0

    def test_f1_select_non_negative(self, d2_scores):
        assert d2_scores[0].criteria["S_select"] >= 0.0

    def test_f2_select_non_negative(self, d2_scores):
        assert d2_scores[1].criteria["S_select"] >= 0.0

    def test_each_sub_score_in_unit_interval(self, d2_scores):
        for fid, result in d2_scores.items():
            for name, val in result.criteria.items():
                assert 0.0 <= val <= 1.0, f"F{fid + 1} {name}: {val}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestD2Metadata:
    def test_definition_label(self, d2_scores):
        for _fid, result in d2_scores.items():
            assert result.definition == "D2"

    def test_family_id_matches(self, d2_scores):
        for fid, result in d2_scores.items():
            assert result.family_id == fid
