"""Tests for analyze_agreement — pairwise agreement statistics.

Tests cover:
- Cohen's κ: perfect agreement → 1.0, random → ≈0
- Spearman ρ: monotonic → 1.0
- Disagreement characterization: direction detection
- Integration: run on mode_b_run, verify JSON schema
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self):
        from analyze_agreement import cohens_kappa

        a = [True, True, False, False, True]
        b = [True, True, False, False, True]
        assert cohens_kappa(a, b) == pytest.approx(1.0)

    def test_perfect_disagreement(self):
        from analyze_agreement import cohens_kappa

        a = [True, True, False, False]
        b = [False, False, True, True]
        assert cohens_kappa(a, b) == pytest.approx(-1.0)

    def test_random_near_zero(self):
        from analyze_agreement import cohens_kappa

        rng = np.random.default_rng(42)
        a = rng.choice([True, False], size=1000).tolist()
        b = rng.choice([True, False], size=1000).tolist()
        kappa = cohens_kappa(a, b)
        assert -0.1 < kappa < 0.1

    def test_all_same_label(self):
        """When both raters always say True, kappa is undefined (0 by convention)."""
        from analyze_agreement import cohens_kappa

        a = [True, True, True]
        b = [True, True, True]
        assert cohens_kappa(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pairwise agreement computation
# ---------------------------------------------------------------------------


class TestPairwiseAgreement:
    def test_spearman_monotonic(self):
        from analyze_agreement import compute_pairwise

        scores_i = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores_j = [0.2, 0.4, 0.6, 0.8, 1.0]
        passes_i = [False, False, True, True, True]
        passes_j = [False, True, True, True, True]

        result = compute_pairwise(scores_i, scores_j, passes_i, passes_j)
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert 0.0 <= result["percent_agreement"] <= 1.0
        assert "cohens_kappa" in result

    def test_pairwise_keys(self):
        from analyze_agreement import compute_pairwise

        result = compute_pairwise(
            [0.1, 0.5, 0.9],
            [0.2, 0.6, 0.8],
            [False, True, True],
            [False, True, True],
        )
        expected_keys = {"cohens_kappa", "spearman_rho", "percent_agreement"}
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Disagreement characterization
# ---------------------------------------------------------------------------


class TestDisagreementCharacterization:
    def test_direction_detection(self):
        from analyze_agreement import characterize_disagreements

        # D_i accepts, D_j rejects for some rows
        rows = [
            {"regime": "E1", "family_id": 0, "Di_pass": True, "Dj_pass": False},
            {"regime": "E1", "family_id": 1, "Di_pass": False, "Dj_pass": True},
            {"regime": "E1", "family_id": 2, "Di_pass": True, "Dj_pass": True},
            {"regime": "E2", "family_id": 0, "Di_pass": True, "Dj_pass": False},
        ]
        result = characterize_disagreements(rows)
        assert result["i_accepts_j_rejects"] == 2
        assert result["j_accepts_i_rejects"] == 1
        assert result["total_disagreements"] == 3

    def test_by_regime(self):
        from analyze_agreement import characterize_disagreements

        rows = [
            {"regime": "E1", "family_id": 0, "Di_pass": True, "Dj_pass": False},
            {"regime": "E2", "family_id": 0, "Di_pass": True, "Dj_pass": False},
            {"regime": "E2", "family_id": 1, "Di_pass": False, "Dj_pass": True},
        ]
        result = characterize_disagreements(rows)
        assert result["by_regime"]["E1"]["total"] == 1
        assert result["by_regime"]["E2"]["total"] == 2

    def test_by_family(self):
        from analyze_agreement import characterize_disagreements

        rows = [
            {"regime": "E1", "family_id": 0, "Di_pass": True, "Dj_pass": False},
            {"regime": "E1", "family_id": 2, "Di_pass": False, "Dj_pass": True},
            {"regime": "E1", "family_id": 2, "Di_pass": True, "Dj_pass": False},
        ]
        result = characterize_disagreements(rows)
        assert result["by_family"][0]["total"] == 1
        assert result["by_family"][2]["total"] == 2


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------


class TestFleissKappa:
    def test_perfect_agreement(self):
        from analyze_agreement import fleiss_kappa

        # 4 raters all agree on each item
        # ratings[i] = counts of each category for item i
        # 3 items, 2 categories, 4 raters all agree
        ratings = np.array(
            [
                [4, 0],  # all say category 0
                [0, 4],  # all say category 1
                [4, 0],  # all say category 0
            ]
        )
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_random_near_zero(self):
        from analyze_agreement import fleiss_kappa

        rng = np.random.default_rng(42)
        # 100 items, 2 categories, 4 raters — random assignments
        ratings = np.zeros((100, 2), dtype=int)
        for i in range(100):
            choices = rng.choice(2, size=4)
            ratings[i, 0] = np.sum(choices == 0)
            ratings[i, 1] = np.sum(choices == 1)
        kappa = fleiss_kappa(ratings)
        assert -0.15 < kappa < 0.15


# ---------------------------------------------------------------------------
# Integration: full agreement analysis on mode_b_run
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_analyze_run_produces_valid_json(self, mode_b_run):
        """Score one run, then analyze agreement among D1–D4."""
        from score_benchmark import score_run

        rows = score_run(mode_b_run, regime="E1", seed=42)
        # Build score matrix format expected by analyze
        from analyze_agreement import analyze_agreement

        result = analyze_agreement(rows)
        # Should have pairwise section
        assert "pairwise" in result
        # 6 pairs: C(4,2)
        assert len(result["pairwise"]) == 6
        # Each pair has required keys
        for _pair_key, pair_data in result["pairwise"].items():
            assert "cohens_kappa" in pair_data
            assert "spearman_rho" in pair_data

    def test_d2_d3_graded_divergence_on_f3(self, mode_b_run):
        """D2 and D3 give F3 different graded scores (both may reject at threshold)."""
        from score_benchmark import score_run

        rows = score_run(mode_b_run, regime="E1", seed=42)
        # F3 (family_id=2): D2 should score ~0, D3 should score higher
        f3 = [r for r in rows if r["family_id"] == 2][0]
        assert f3["D3_score"] > f3["D2_score"]

        from analyze_agreement import analyze_agreement

        result = analyze_agreement(rows)
        d2_d3 = result["pairwise"].get("D2_D3")
        assert d2_d3 is not None
        # Spearman ρ should be computable (not NaN)
        assert not np.isnan(d2_d3["spearman_rho"])
