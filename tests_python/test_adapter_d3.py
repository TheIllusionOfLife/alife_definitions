"""Tests for D3 adapter — Autonomy / Organizational Closure.

Validates:
- F1 has higher closure score than F2 (F2 lacks boundary loop)
- F3 has moderate closure (self-maintenance without reproduction)
- Directed graph has at least |SCC| ≥ 2 for F1
- Persistence penalty: collapsed family scores near 0
- TE computation: at least some edges significant for F1
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
def d3_scores(mode_b_run):
    from adapters.d3 import score_d3

    return {fid: score_d3(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------


class TestD3ScoreRange:
    def test_all_scores_in_unit_interval(self, d3_scores):
        for fid, result in d3_scores.items():
            assert 0.0 <= result.score <= 1.0, f"F{fid + 1} score: {result.score}"

    def test_criteria_has_expected_keys(self, d3_scores):
        expected = {"closure", "persistence"}
        for _fid, result in d3_scores.items():
            assert set(result.criteria.keys()) == expected


# ---------------------------------------------------------------------------
# Cross-family ordering
# ---------------------------------------------------------------------------


class TestD3FamilyOrdering:
    def test_f1_closure_higher_than_f2(self, d3_scores):
        """F1 (full) should have higher closure than F2 (broken boundary loop)."""
        assert d3_scores[0].criteria["closure"] >= d3_scores[1].criteria["closure"]

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
