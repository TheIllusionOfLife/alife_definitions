"""Tests for score_benchmark — score matrix generation.

Tests cover:
- TSV output has expected columns for all families and definitions
- Scores match direct adapters.score_all() output
- CLI seed/regime parsing
"""

from __future__ import annotations

import csv
import io

import pytest


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "regime",
    "seed",
    "family_id",
    "D1_score",
    "D1_pass",
    "D2_score",
    "D2_pass",
    "D3_score",
    "D3_pass",
    "D4_score",
    "D4_pass",
]

# Sub-criterion columns that should also be present
D1_SUB_COLUMNS = [
    "D1_metabolism",
    "D1_boundary",
    "D1_homeostasis",
    "D1_response",
    "D1_reproduction",
    "D1_evolution",
    "D1_growth",
]

D2_SUB_COLUMNS = ["D2_S_reprod", "D2_S_hered", "D2_S_select"]
D3_SUB_COLUMNS = ["D3_closure", "D3_persistence"]
D4_SUB_COLUMNS = ["D4_S_info_present", "D4_S_info_causal", "D4_S_info_preserved"]


# ---------------------------------------------------------------------------
# Unit tests: build_score_row
# ---------------------------------------------------------------------------


class TestBuildScoreRow:
    def test_row_has_all_columns(self, mode_b_run):
        from score_benchmark import build_score_row

        row = build_score_row(mode_b_run, regime="E1", seed=42, family_id=0)
        for col in EXPECTED_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_row_has_sub_criteria(self, mode_b_run):
        from score_benchmark import build_score_row

        row = build_score_row(mode_b_run, regime="E1", seed=42, family_id=0)
        for col in D1_SUB_COLUMNS + D2_SUB_COLUMNS + D3_SUB_COLUMNS + D4_SUB_COLUMNS:
            assert col in row, f"Missing sub-criterion column: {col}"

    def test_scores_in_unit_interval(self, mode_b_run):
        from score_benchmark import build_score_row

        row = build_score_row(mode_b_run, regime="E1", seed=42, family_id=0)
        for defn in ["D1", "D2", "D3", "D4"]:
            score = row[f"{defn}_score"]
            assert 0.0 <= score <= 1.0, f"{defn}_score = {score}"

    def test_pass_is_bool(self, mode_b_run):
        from score_benchmark import build_score_row

        row = build_score_row(mode_b_run, regime="E1", seed=42, family_id=0)
        for defn in ["D1", "D2", "D3", "D4"]:
            assert isinstance(row[f"{defn}_pass"], bool)


# ---------------------------------------------------------------------------
# Unit tests: scores match adapters.score_all()
# ---------------------------------------------------------------------------


class TestScoresMatchAdapters:
    def test_scores_match_score_all(self, mode_b_run):
        from adapters import score_all
        from score_benchmark import build_score_row

        for fid in [0, 1, 2]:
            row = build_score_row(mode_b_run, regime="E1", seed=42, family_id=fid)
            direct = score_all(mode_b_run, family_id=fid)
            for defn in ["D1", "D2", "D3", "D4"]:
                assert row[f"{defn}_score"] == pytest.approx(
                    direct[defn].score, abs=1e-9
                ), f"F{fid}/{defn} mismatch"


# ---------------------------------------------------------------------------
# Integration: score_run produces all families
# ---------------------------------------------------------------------------


class TestScoreRun:
    def test_score_run_returns_all_families(self, mode_b_run):
        from score_benchmark import score_run

        rows = score_run(mode_b_run, regime="E1", seed=42)
        family_ids = {r["family_id"] for r in rows}
        assert family_ids == {0, 1, 2}

    def test_score_run_row_count(self, mode_b_run):
        from score_benchmark import score_run

        rows = score_run(mode_b_run, regime="E1", seed=42)
        assert len(rows) == 3  # 3 families


# ---------------------------------------------------------------------------
# TSV formatting
# ---------------------------------------------------------------------------


class TestTsvOutput:
    def test_format_tsv(self, mode_b_run):
        from score_benchmark import format_tsv, score_run

        rows = score_run(mode_b_run, regime="E1", seed=42)
        tsv = format_tsv(rows)
        reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
        parsed = list(reader)
        assert len(parsed) == 3
        for col in EXPECTED_COLUMNS:
            assert col in parsed[0], f"Missing TSV column: {col}"
