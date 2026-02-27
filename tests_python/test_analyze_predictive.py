"""Tests for analyze_predictive — predictive validity analysis.

Tests cover:
- Threshold sweep on synthetic data produces reasonable threshold
- ROC-AUC on perfect predictor → 1.0
- Balanced accuracy computation
- Sensitivity sweep: ±20% threshold changes ≤ bounded accuracy shift
- alive_count_AUC extraction for families
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Balanced accuracy
# ---------------------------------------------------------------------------


class TestBalancedAccuracy:
    def test_perfect_prediction(self):
        from analyze_predictive import balanced_accuracy

        y_true = [True, True, False, False]
        y_pred = [True, True, False, False]
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        from analyze_predictive import balanced_accuracy

        y_true = [True, True, False, False]
        y_pred = [False, False, True, True]
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_random_near_half(self):
        from analyze_predictive import balanced_accuracy

        # Always predict True
        y_true = [True, True, False, False]
        y_pred = [True, True, True, True]
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------------------------


class TestRocAuc:
    def test_perfect_predictor(self):
        from analyze_predictive import roc_auc_score

        y_true = [True, True, True, False, False, False]
        y_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        assert roc_auc_score(y_true, y_scores) == pytest.approx(1.0)

    def test_worst_predictor(self):
        from analyze_predictive import roc_auc_score

        y_true = [True, True, True, False, False, False]
        y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        assert roc_auc_score(y_true, y_scores) == pytest.approx(0.0)

    def test_random_near_half(self):
        from analyze_predictive import roc_auc_score

        rng = np.random.default_rng(42)
        y_true = [True] * 50 + [False] * 50
        y_scores = rng.uniform(0, 1, 100).tolist()
        auc = roc_auc_score(y_true, y_scores)
        assert 0.3 < auc < 0.7

    def test_single_class_returns_nan(self):
        from analyze_predictive import roc_auc_score

        y_true = [True, True, True]
        y_scores = [0.9, 0.8, 0.7]
        assert np.isnan(roc_auc_score(y_true, y_scores))


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


class TestThresholdSweep:
    def test_sweep_finds_optimal(self):
        from analyze_predictive import sweep_threshold

        # Perfect separation: scores > 0.5 are "alive"
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [False, False, False, True, True, True]
        best_thresh, best_ba = sweep_threshold(scores, labels)
        assert best_ba == pytest.approx(1.0)
        assert 0.3 < best_thresh < 0.7

    def test_sweep_worst_case(self):
        from analyze_predictive import sweep_threshold

        # Anti-correlated: high score → not alive
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [False, False, False, True, True, True]
        best_thresh, best_ba = sweep_threshold(scores, labels)
        # Should still find the best threshold it can
        assert 0.0 <= best_ba <= 1.0
        assert 0.0 <= best_thresh <= 1.0


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_sensitivity_sweep(self):
        from analyze_predictive import sensitivity_sweep

        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [False, False, False, True, True, True]
        threshold = 0.5
        result = sensitivity_sweep(scores, labels, threshold, delta=0.2)
        assert "ba_at_threshold" in result
        assert "ba_minus" in result
        assert "ba_plus" in result
        assert "max_ba_change" in result


# ---------------------------------------------------------------------------
# Family alive_count AUC extraction
# ---------------------------------------------------------------------------


class TestFamilyAliveAuc:
    def test_extract_family_alive_auc(self, mode_b_run):
        from analyze_predictive import extract_family_alive_auc

        auc = extract_family_alive_auc(mode_b_run, family_id=0, tail_fraction=0.3)
        assert auc > 0  # F1 should persist

    def test_zero_tail_fraction(self, mode_b_run):
        from analyze_predictive import extract_family_alive_auc

        # With tail_fraction=1.0, uses entire series
        auc_full = extract_family_alive_auc(mode_b_run, family_id=0, tail_fraction=1.0)
        auc_tail = extract_family_alive_auc(mode_b_run, family_id=0, tail_fraction=0.3)
        # Full AUC should be >= tail AUC (more area)
        assert auc_full >= auc_tail


# ---------------------------------------------------------------------------
# Integration: calibrate + evaluate pipeline
# ---------------------------------------------------------------------------


class TestCalibratePipeline:
    def test_calibrate_single_definition(self, mode_b_run):
        """Calibrate threshold for D1 on a single run, then evaluate."""
        from analyze_predictive import calibrate_definition, evaluate_definition

        # Use the same run as both calibration and test (smoke test only)
        cal_data = [{"run": mode_b_run, "regime": "E1", "seed": 42}]

        thresh = calibrate_definition("D1", cal_data, tail_fraction=0.3)
        assert 0.0 <= thresh <= 1.0

        # Evaluate on same data (smoke test)
        metrics = evaluate_definition("D1", cal_data, thresh)
        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
