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

    Groups tied scores so that the ROC curve steps correctly when
    multiple samples share the same score value.

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

    # Group by unique score to handle ties correctly
    i = 0
    while i < len(pairs):
        # Advance through all samples with the same score
        current_score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == current_score:
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
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
    """Evaluate balanced accuracy at threshold and ±delta (additive)."""
    preds_at = [s >= threshold for s in scores]
    ba_at = balanced_accuracy(labels, preds_at)

    thresh_minus = max(0.0, threshold - delta)
    preds_minus = [s >= thresh_minus for s in scores]
    ba_minus = balanced_accuracy(labels, preds_minus)

    thresh_plus = min(1.0, threshold + delta)
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
# Recovery time
# ---------------------------------------------------------------------------


def extract_recovery_time(
    run_summary: dict,
    family_id: int,
    dip_threshold: float = 0.5,
) -> float:
    """Steps to recover to 80% of initial alive_count after deepest dip.

    Returns normalized [0, 1]: 1 = instant recovery, 0 = never recovered.
    """
    series = extract_family_series(run_summary, family_id)
    alive = series["alive_count"]

    if len(alive) < 4:
        return 0.0

    initial = float(np.mean(alive[: max(1, len(alive) // 10)]))
    if initial <= 0:
        return 0.0

    # Find deepest dip below threshold fraction of initial
    threshold_val = initial * dip_threshold
    dip_idx = -1
    min_val = initial
    for i, v in enumerate(alive):
        if v < min_val:
            min_val = v
            if v < threshold_val:
                dip_idx = i

    if dip_idx < 0:
        # No significant dip — population was stable
        return 1.0

    # Find recovery point: first step after dip where alive >= 80% of initial
    recovery_target = initial * 0.8
    for j in range(dip_idx, len(alive)):
        if alive[j] >= recovery_target:
            # Normalize: faster recovery → higher score
            recovery_steps = j - dip_idx
            max_possible = len(alive) - dip_idx
            return float(np.clip(1.0 - recovery_steps / max_possible, 0.0, 1.0))

    return 0.0  # Never recovered


# ---------------------------------------------------------------------------
# Lineage diversity
# ---------------------------------------------------------------------------


def extract_lineage_diversity(
    run_summary: dict,
    family_id: int,
    tail_fraction: float = 0.2,
) -> float:
    """Parent reproductive diversity in the tail of the lineage.

    Computes the ratio of unique reproducing parents to total birth events
    in the final ``tail_fraction`` of lineage events.  A high ratio means
    many distinct genotypes are successfully reproducing (maintained
    genetic diversity); a low ratio means one or few genotypes dominate
    reproduction.

    Returns a value in [0, 1].  0.0 when no lineage events exist.
    """
    from adapters.common import extract_family_lineage

    lineage = extract_family_lineage(run_summary, family_id)

    if not lineage:
        return 0.0

    # Take the tail fraction of lineage events
    n = len(lineage)
    start = max(0, n - int(n * tail_fraction))
    tail = lineage[start:]

    if not tail:
        return 0.0

    events_with_parent = [e for e in tail if "parent_genome_hash" in e]
    n_events = len(events_with_parent)

    if n_events == 0:
        return 0.0

    parent_hashes = set(e["parent_genome_hash"] for e in events_with_parent)

    # Ratio: unique parents / total births.
    # 1.0 = every birth from a different parent genotype (high diversity).
    # Near 0 = one genotype dominates reproduction (low diversity / clonal).
    return float(np.clip(len(parent_hashes) / n_events, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Target extractor registry
# ---------------------------------------------------------------------------

TARGET_EXTRACTORS = {
    "alive_auc": extract_family_alive_auc,
    "recovery_time": extract_recovery_time,
    "lineage_diversity": extract_lineage_diversity,
}


# ---------------------------------------------------------------------------
# Calibration and evaluation
# ---------------------------------------------------------------------------


def _precompute_all_scores(
    data: list[dict],
    tail_fraction: float,
    target: str = "alive_auc",
) -> tuple[dict[str, list[float]], list[float]]:
    """Score all runs once and return per-definition scores and target values.

    Calls ``score_all()`` once per (run, family) instead of once per
    (definition, run, family), avoiding 4x redundant computation.

    Args:
        target: Target extractor name from TARGET_EXTRACTORS.

    Returns:
        Tuple of (scores_by_defn, target_values) where scores_by_defn maps
        each definition name to a list of scores aligned with target_values.
    """
    extractor = TARGET_EXTRACTORS[target]
    scores_by_defn: dict[str, list[float]] = {d: [] for d in DEFINITIONS}
    target_values: list[float] = []

    for entry in data:
        run = entry["run"]
        family_ids = discover_family_ids(run)
        for fid in family_ids:
            result = score_all(run, family_id=fid)
            for d in DEFINITIONS:
                scores_by_defn[d].append(result[d].score)
            # alive_auc uses tail_fraction; others use their own defaults
            if target == "alive_auc":
                val = extractor(run, fid, tail_fraction)
            else:
                val = extractor(run, fid)
            target_values.append(val)

    return scores_by_defn, target_values


def _make_labels(aucs: list[float]) -> tuple[list[bool], float]:
    """Convert alive AUCs to binary labels using median as threshold.

    When all AUCs are identical (e.g. all organisms die), the median
    split produces a single-class target. In this case we log a warning
    and return all-False labels — callers should check for single-class
    before computing metrics.
    """
    if not aucs:
        return [], 0.0
    median_auc = float(np.median(aucs))
    labels = [a > median_auc for a in aucs]
    if len(set(labels)) <= 1:
        import sys

        print(
            f"WARNING: degenerate labels — all AUCs equal ({median_auc:.4f}), "
            "single-class target produced",
            file=sys.stderr,
        )
    return labels, median_auc


def calibrate_definition(
    defn: str,
    cal_data: list[dict],
    tail_fraction: float = 0.3,
    target: str = "alive_auc",
) -> float:
    """Calibrate threshold for a single definition on calibration data.

    Args:
        defn: Definition name ("D1", "D2", etc.).
        cal_data: List of {"run": dict, "regime": str, "seed": int}.
        tail_fraction: Fraction of run tail for alive_count AUC.
        target: Prediction target from TARGET_EXTRACTORS.

    Returns:
        Optimal threshold maximizing balanced accuracy.
    """
    scores_by_defn, aucs = _precompute_all_scores(cal_data, tail_fraction, target=target)
    scores = scores_by_defn[defn]
    if not scores:
        return 0.5
    labels, _median = _make_labels(aucs)
    thresh, _ba = sweep_threshold(scores, labels)
    return thresh


def evaluate_definition(
    defn: str,
    test_data: list[dict],
    threshold: float,
    tail_fraction: float = 0.3,
    target: str = "alive_auc",
) -> dict:
    """Evaluate a definition on test data with a frozen threshold.

    Returns dict with roc_auc, precision, recall, balanced_accuracy.
    """
    scores_by_defn, aucs = _precompute_all_scores(test_data, tail_fraction, target=target)
    scores = scores_by_defn[defn]
    labels, _median = _make_labels(aucs)

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
    """CLI entry point: calibrate thresholds and evaluate predictive validity."""
    parser = argparse.ArgumentParser(description="Predictive validity analysis")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--cal-seeds", default="0-99", help="Calibration seed range")
    parser.add_argument("--test-seeds", default="100-199", help="Test seed range")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5", help="Comma-separated regimes")
    parser.add_argument(
        "--target",
        default="alive_auc",
        choices=list(TARGET_EXTRACTORS.keys()),
        help="Prediction target (default: alive_auc)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON (default: stdout)")
    args = parser.parse_args()

    from experiment_common import log, parse_regimes, parse_seed_range, safe_path

    data_dir = args.data_dir.resolve()
    cal_seeds = parse_seed_range(args.cal_seeds)
    test_seeds = parse_seed_range(args.test_seeds)
    regimes = parse_regimes(args.regimes)

    def load_runs(seeds: list[int]) -> list[dict]:
        """Load benchmark run JSONs for given seeds across all regimes."""
        runs = []
        for regime in regimes:
            regime_dir = safe_path(data_dir, regime)
            for seed in seeds:
                path = regime_dir / f"seed_{seed:03d}.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    run = json.load(f)
                runs.append({"run": run, "regime": regime, "seed": seed})
        return runs

    cal_data = load_runs(cal_seeds)
    test_data = load_runs(test_seeds)
    log(f"Loaded {len(cal_data)} calibration runs, {len(test_data)} test runs")

    results = {"definitions": {}, "frozen_thresholds": {}}

    for defn in DEFINITIONS:
        log(f"Calibrating {defn}...")
        thresh = calibrate_definition(defn, cal_data, target=args.target)
        results["frozen_thresholds"][defn] = thresh
        log(f"  {defn} threshold: {thresh:.3f}")

        if test_data:
            log(f"Evaluating {defn} on test set...")
            metrics = evaluate_definition(defn, test_data, thresh, target=args.target)
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
