"""Predictive validity analysis for D1–D4 definition scores.

Calibrates thresholds on calibration seeds (0–99), evaluates on test seeds
(100–199). Reports ROC-AUC, precision/recall at frozen threshold, and
sensitivity analysis.

Usage:
    uv run python scripts/analyze_predictive.py experiments/benchmark/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from adapters import score_all
from adapters.common import discover_family_ids, extract_family_series

DEFINITIONS = ["D1", "D2", "D3", "D4"]

# np.trapezoid was added in NumPy 2.0; np.trapz was removed in NumPy 2.0.
try:
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Balanced accuracy
# ---------------------------------------------------------------------------


def balanced_accuracy(y_true: list[bool], y_pred: list[bool]) -> float:
    """Compute balanced accuracy = (sensitivity + specificity) / 2."""
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sensitivity + specificity) / 2.0


# ---------------------------------------------------------------------------
# ROC-AUC (manual implementation)
# ---------------------------------------------------------------------------


def roc_auc_score(y_true: list[bool], y_scores: list[float]) -> float:
    """Compute ROC-AUC using the trapezoidal rule.

    Returns NaN if only one class is present.
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by score descending
    pairs = sorted(zip(y_scores, y_true, strict=True), key=lambda x: -x[0])

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for _score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return float(_trapezoid(tpr_list, fpr_list))


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def sweep_threshold(
    scores: list[float],
    labels: list[bool],
    n_steps: int = 100,
) -> tuple[float, float]:
    """Sweep thresholds to maximize balanced accuracy.

    Returns (best_threshold, best_balanced_accuracy).
    """
    best_thresh = 0.0
    best_ba = 0.0

    for i in range(n_steps + 1):
        thresh = i / n_steps
        preds = [s >= thresh for s in scores]
        ba = balanced_accuracy(labels, preds)
        if ba > best_ba:
            best_ba = ba
            best_thresh = thresh

    return best_thresh, best_ba


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_sweep(
    scores: list[float],
    labels: list[bool],
    threshold: float,
    delta: float = 0.2,
) -> dict:
    """Evaluate balanced accuracy at threshold and ±delta."""
    preds_at = [s >= threshold for s in scores]
    ba_at = balanced_accuracy(labels, preds_at)

    thresh_minus = max(0.0, threshold * (1.0 - delta))
    preds_minus = [s >= thresh_minus for s in scores]
    ba_minus = balanced_accuracy(labels, preds_minus)

    thresh_plus = min(1.0, threshold * (1.0 + delta))
    preds_plus = [s >= thresh_plus for s in scores]
    ba_plus = balanced_accuracy(labels, preds_plus)

    return {
        "threshold": threshold,
        "ba_at_threshold": ba_at,
        "threshold_minus": thresh_minus,
        "ba_minus": ba_minus,
        "threshold_plus": thresh_plus,
        "ba_plus": ba_plus,
        "max_ba_change": max(abs(ba_at - ba_minus), abs(ba_at - ba_plus)),
    }


# ---------------------------------------------------------------------------
# Family alive_count AUC extraction
# ---------------------------------------------------------------------------


def extract_family_alive_auc(
    run_summary: dict,
    family_id: int,
    tail_fraction: float = 0.3,
) -> float:
    """Compute AUC of alive_count for a family over the tail of the run.

    Args:
        run_summary: Parsed JSON run output.
        family_id: Family index.
        tail_fraction: Fraction of the run to use (from the end).

    Returns:
        AUC value (trapezoidal rule on the tail segment).
    """
    series = extract_family_series(run_summary, family_id)
    alive = series["alive_count"]

    if len(alive) < 2:
        return 0.0

    n = len(alive)
    start = max(0, n - int(n * tail_fraction))
    tail = alive[start:]

    if len(tail) < 2:
        return 0.0

    # Use integer indices as x-axis
    x = np.arange(len(tail))
    return float(_trapezoid(tail, x))


# ---------------------------------------------------------------------------
# Calibration and evaluation
# ---------------------------------------------------------------------------


def _collect_scores_and_labels(
    defn: str,
    data: list[dict],
    tail_fraction: float,
) -> tuple[list[float], list[bool], float]:
    """Score runs and compute alive labels. Returns (scores, labels, median_auc)."""
    scores: list[float] = []
    aucs: list[float] = []

    for entry in data:
        run = entry["run"]
        family_ids = discover_family_ids(run)
        for fid in family_ids:
            result = score_all(run, family_id=fid)
            scores.append(result[defn].score)
            auc = extract_family_alive_auc(run, fid, tail_fraction)
            aucs.append(auc)

    median_auc = float(np.median(aucs)) if aucs else 0.0
    labels = [a > median_auc for a in aucs]
    return scores, labels, median_auc


def calibrate_definition(
    defn: str,
    cal_data: list[dict],
    tail_fraction: float = 0.3,
) -> float:
    """Calibrate threshold for a single definition on calibration data.

    Args:
        defn: Definition name ("D1", "D2", etc.).
        cal_data: List of {"run": dict, "regime": str, "seed": int}.
        tail_fraction: Fraction of run tail for alive_count AUC.

    Returns:
        Optimal threshold maximizing balanced accuracy.
    """
    scores, labels, _median = _collect_scores_and_labels(defn, cal_data, tail_fraction)
    if not scores:
        return 0.5
    thresh, _ba = sweep_threshold(scores, labels)
    return thresh


def evaluate_definition(
    defn: str,
    test_data: list[dict],
    threshold: float,
    tail_fraction: float = 0.3,
) -> dict:
    """Evaluate a definition on test data with a frozen threshold.

    Returns dict with roc_auc, precision, recall, balanced_accuracy.
    """
    scores, labels, _median = _collect_scores_and_labels(defn, test_data, tail_fraction)

    if not scores:
        return {
            "roc_auc": float("nan"),
            "precision": 0.0,
            "recall": 0.0,
            "balanced_accuracy": 0.0,
        }

    auc = roc_auc_score(labels, scores)

    preds = [s >= threshold for s in scores]
    tp = sum(1 for t, p in zip(labels, preds, strict=True) if t and p)
    fp = sum(1 for t, p in zip(labels, preds, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(labels, preds, strict=True) if t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ba = balanced_accuracy(labels, preds)

    sensitivity_result = sensitivity_sweep(scores, labels, threshold)

    return {
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_accuracy": float(ba),
        "threshold": float(threshold),
        "sensitivity": sensitivity_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive validity analysis")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--cal-seeds", default="0-99", help="Calibration seed range")
    parser.add_argument("--test-seeds", default="100-199", help="Test seed range")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5", help="Comma-separated regimes")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON (default: stdout)")
    args = parser.parse_args()

    from experiment_common import log, safe_path
    from score_benchmark import _parse_regimes, _parse_seed_range

    data_dir = args.data_dir.resolve()
    cal_seeds = _parse_seed_range(args.cal_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    regimes = _parse_regimes(args.regimes)

    def load_runs(seeds: list[int]) -> list[dict]:
        runs = []
        for regime in regimes:
            regime_dir = safe_path(data_dir, regime)
            for seed in seeds:
                path = regime_dir / f"seed_{seed:03d}.json"
                if not path.exists():
                    continue
                import json as _json

                with open(path) as f:
                    run = _json.load(f)
                runs.append({"run": run, "regime": regime, "seed": seed})
        return runs

    cal_data = load_runs(cal_seeds)
    test_data = load_runs(test_seeds)
    log(f"Loaded {len(cal_data)} calibration runs, {len(test_data)} test runs")

    results = {"definitions": {}, "frozen_thresholds": {}}

    for defn in DEFINITIONS:
        log(f"Calibrating {defn}...")
        thresh = calibrate_definition(defn, cal_data)
        results["frozen_thresholds"][defn] = thresh
        log(f"  {defn} threshold: {thresh:.3f}")

        if test_data:
            log(f"Evaluating {defn} on test set...")
            metrics = evaluate_definition(defn, test_data, thresh)
            results["definitions"][defn] = metrics
            log(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        else:
            log("  No test data — skipping evaluation")

    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
