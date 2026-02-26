"""Tests for D4 adapter — Information Maintenance.

Validates:
- F1 and F2 score higher than F3
- S_info_preserved ≈ 0 for F3 (no lineage events)
- S_info_causal is non-negative
- TE(genome_diversity → alive_count) computable for F1/F2
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
def d4_scores(mode_b_run):
    from adapters.d4 import score_d4

    return {fid: score_d4(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------


class TestD4ScoreRange:
    def test_all_scores_in_unit_interval(self, d4_scores):
        for fid, result in d4_scores.items():
            assert 0.0 <= result.score <= 1.0, f"F{fid + 1} score: {result.score}"

    def test_criteria_has_expected_keys(self, d4_scores):
        expected = {"S_info_present", "S_info_causal", "S_info_preserved"}
        for _fid, result in d4_scores.items():
            assert set(result.criteria.keys()) == expected


# ---------------------------------------------------------------------------
# Cross-family ordering
# ---------------------------------------------------------------------------


class TestD4FamilyOrdering:
    def test_f1_higher_than_f3(self, d4_scores):
        assert d4_scores[0].score > d4_scores[2].score

    def test_f2_higher_than_f3(self, d4_scores):
        assert d4_scores[1].score > d4_scores[2].score


# ---------------------------------------------------------------------------
# Sub-score properties
# ---------------------------------------------------------------------------


class TestD4SubScores:
    def test_f3_preserved_zero(self, d4_scores):
        """F3 has no lineage → S_info_preserved should be 0."""
        assert d4_scores[2].criteria["S_info_preserved"] == 0.0

    def test_causal_non_negative(self, d4_scores):
        for _fid, result in d4_scores.items():
            assert result.criteria["S_info_causal"] >= 0.0

    def test_each_sub_score_in_unit_interval(self, d4_scores):
        for fid, result in d4_scores.items():
            for name, val in result.criteria.items():
                assert 0.0 <= val <= 1.0, f"F{fid + 1} {name}: {val}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestD4Metadata:
    def test_definition_label(self, d4_scores):
        for _fid, result in d4_scores.items():
            assert result.definition == "D4"

    def test_family_id_matches(self, d4_scores):
        for fid, result in d4_scores.items():
            assert result.family_id == fid
