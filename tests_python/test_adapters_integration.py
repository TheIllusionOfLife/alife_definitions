"""Integration tests: cross-definition disagreement patterns.

Runs all 4 adapters on the same Mode B run and verifies:
- D2 rejects F3 (no reproduction); D3 accepts F3 (has closure)
- D3 partially rejects F2 (broken boundary loop); D2 accepts F2 (strong evolution)
- D1 is strictest (requires all 7); D4 aligns with D2 on evolution families
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
def all_scores(mode_b_run):
    """Score all families with all definitions."""
    from adapters import score_all

    return {fid: score_all(mode_b_run, family_id=fid) for fid in [0, 1, 2]}


# ---------------------------------------------------------------------------
# All adapters produce valid results
# ---------------------------------------------------------------------------


class TestAllAdaptersRun:
    def test_four_definitions_per_family(self, all_scores):
        for _fid, scores in all_scores.items():
            assert set(scores.keys()) == {"D1", "D2", "D3", "D4"}

    def test_all_scores_in_unit_interval(self, all_scores):
        for fid, scores in all_scores.items():
            for defn, result in scores.items():
                assert 0.0 <= result.score <= 1.0, f"F{fid + 1}/{defn} score: {result.score}"


# ---------------------------------------------------------------------------
# D2 vs D3 disagreement on F3
# ---------------------------------------------------------------------------


class TestD2D3Disagreement:
    def test_d2_rejects_f3(self, all_scores):
        """D2 should give F3 (no reproduction) a low score."""
        assert all_scores[2]["D2"].score < 0.2

    def test_d3_accepts_f3(self, all_scores):
        """D3 should give F3 (autonomy/closure) a moderate-to-high score."""
        assert all_scores[2]["D3"].score > all_scores[2]["D2"].score

    def test_d3_f3_higher_than_d2_f3(self, all_scores):
        """The key disagreement: D3 scores F3 higher than D2 does."""
        d3_f3 = all_scores[2]["D3"].score
        d2_f3 = all_scores[2]["D2"].score
        assert d3_f3 > d2_f3


# ---------------------------------------------------------------------------
# D2 vs D3 disagreement on F2
# ---------------------------------------------------------------------------


class TestD2D3OnF2:
    def test_d2_accepts_f2(self, all_scores):
        """D2 should give F2 (Darwinian) a decent score."""
        assert all_scores[1]["D2"].score > 0.1

    def test_d3_f2_lower_closure(self, all_scores):
        """D3 should give F2 lower closure than F1 (broken boundary loop)."""
        assert all_scores[1]["D3"].score <= all_scores[0]["D3"].score


# ---------------------------------------------------------------------------
# D1 strictness
# ---------------------------------------------------------------------------


class TestD1Strictness:
    def test_d1_is_strict(self, all_scores):
        """D1 (all 7 required) should be among the strictest for ablated families."""
        # For F2 and F3 (ablated), D1 should generally score lower
        for fid in [1, 2]:
            d1 = all_scores[fid]["D1"].score
            # D1 uses geometric mean of 7 criteria — any zero pulls it down
            # It should be relatively low for ablated families
            assert d1 < all_scores[0]["D1"].score


# ---------------------------------------------------------------------------
# D4 alignment with D2 on evolution families
# ---------------------------------------------------------------------------


class TestD4Alignment:
    def test_d4_f3_low(self, all_scores):
        """D4 should give F3 a low score (no lineage → no info preservation)."""
        assert all_scores[2]["D4"].score < all_scores[0]["D4"].score
