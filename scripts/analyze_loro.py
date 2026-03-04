"""Leave-One-Regime-Out (LORO) cross-validation for definition generalization.

5-fold: hold out 1 regime → calibrate thresholds on other 4 → evaluate on
held-out regime's test seeds. Reports per-fold AUC per definition + mean AUC.

Usage:
    uv run python scripts/analyze_loro.py experiments/benchmark/ \
        --test-seeds 100-199 --regimes E1,E2,E3,E4,E5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from analyze_predictive import (
    TARGET_EXTRACTORS,
    _make_labels,
    _precompute_all_scores,
    calibrate_definition,
    evaluate_definition,
)
from experiment_common import log, parse_regimes, parse_seed_range, safe_path

DEFINITIONS = ["D1", "D2", "D3", "D4"]


def _load_runs(
    data_dir: Path,
    seeds: list[int],
    regimes: list[str],
) -> dict[str, list[dict]]:
    """Load benchmark runs grouped by regime."""
    by_regime: dict[str, list[dict]] = {r: [] for r in regimes}
    for regime in regimes:
        regime_dir = safe_path(data_dir, regime)
        for seed in seeds:
            path = regime_dir / f"seed_{seed:03d}.json"
            if not path.exists():
                continue
            with open(path) as f:
                run = json.load(f)
            by_regime[regime].append({"run": run, "regime": regime, "seed": seed})
    return by_regime


def analyze_loro(
    data_dir: Path,
    seeds: list[int],
    regimes: list[str],
    target: str = "alive_auc",
) -> dict:
    """Run LORO cross-validation."""
    runs_by_regime = _load_runs(data_dir, seeds, regimes)

    per_fold: dict[str, dict] = {}

    for held_out in regimes:
        log(f"LORO fold: held out = {held_out}")

        # Train on all other regimes
        train_data: list[dict] = []
        for r in regimes:
            if r != held_out:
                train_data.extend(runs_by_regime[r])

        test_data = runs_by_regime[held_out]

        if not train_data or not test_data:
            log(f"  SKIP: insufficient data (train={len(train_data)}, test={len(test_data)})")
            continue

        fold_results: dict[str, dict] = {}
        train_scores, train_targets = _precompute_all_scores(
            train_data, 0.3, target=target, evaluation_mode="legacy"
        )
        train_labels, _ = _make_labels(train_targets)
        test_scores, test_targets = _precompute_all_scores(
            test_data, 0.3, target=target, evaluation_mode="legacy"
        )
        test_labels, _ = _make_labels(test_targets)
        for defn in DEFINITIONS:
            thresh = calibrate_definition(defn, train_scores, train_labels)
            metrics = evaluate_definition(defn, test_scores, test_labels, thresh)
            fold_results[defn] = {
                "roc_auc": round(metrics["roc_auc"], 4),
                "threshold": round(thresh, 4),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            }
            log(f"  {defn}: thresh={thresh:.3f}, AUC={metrics['roc_auc']:.3f}")

        per_fold[held_out] = fold_results

    # Compute mean AUC per definition across folds
    mean_auc: dict[str, float] = {}
    for defn in DEFINITIONS:
        aucs = [
            fold[defn]["roc_auc"]
            for fold in per_fold.values()
            if defn in fold and not np.isnan(fold[defn]["roc_auc"])
        ]
        mean_auc[defn] = round(float(np.mean(aucs)), 4) if aucs else float("nan")

    return {
        "per_fold": per_fold,
        "mean_auc": mean_auc,
        "n_folds": len(per_fold),
        "target": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LORO cross-validation")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--test-seeds", default="100-199", help="Test seed range")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5")
    parser.add_argument(
        "--target",
        default="alive_auc",
        choices=list(TARGET_EXTRACTORS.keys()),
        help="Prediction target",
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    seeds = parse_seed_range(args.test_seeds)
    regimes = parse_regimes(args.regimes)

    log(f"LORO: {len(regimes)} folds, seeds={args.test_seeds}, target={args.target}")

    result = analyze_loro(args.data_dir.resolve(), seeds, regimes, target=args.target)

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log(f"Written to {args.output}")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
