"""Surrogate FPR analysis: measure false positive rate of TE across adapters.

Runs phase-randomized surrogates on actual benchmark data and reports the
fraction that pass significance — a key validation that TE is not producing
spurious edges.

Usage:
    uv run python -m scripts.analyze_surrogate_fpr \
        --data-dir experiments/benchmark --seeds 0-4 --regime E1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from adapters.common import (
    compute_surrogate_fpr,
    discover_family_ids,
    extract_family_series,
)
from experiment_common import log, parse_seed_range, safe_path

# Coupling pairs tested by D1 and D3
_D1_COUPLING_PAIRS = [
    ("energy_mean", "boundary_mean"),
    ("boundary_mean", "energy_mean"),
    ("energy_mean", "waste_mean"),
    ("alive_count", "energy_mean"),
    ("birth_count", "alive_count"),
    ("genome_diversity", "alive_count"),
    ("maturity_mean", "alive_count"),
]

_D3_PROCESS_VARS = ["energy_mean", "waste_mean", "boundary_mean", "birth_count", "maturity_mean"]


def analyze_fpr(
    data_dir: Path,
    seeds: list[int],
    regime: str,
    n_surrogates: int = 100,
) -> dict:
    """Compute surrogate FPR for TE coupling pairs across seeds."""
    results: dict[str, list[float]] = {}

    for seed in seeds:
        seed_file = safe_path(data_dir, regime) / f"seed_{seed:03d}.json"
        if not seed_file.exists():
            log(f"  [skip] {regime}/seed_{seed:03d}.json not found")
            continue

        with open(seed_file) as f:
            run_summary = json.load(f)

        family_ids = discover_family_ids(run_summary)
        if not family_ids:
            continue

        fid = family_ids[0]  # Use first family for FPR analysis
        series = extract_family_series(run_summary, fid)

        # D1 coupling pairs
        for src_name, tgt_name in _D1_COUPLING_PAIRS:
            key = f"D1:{src_name}->{tgt_name}"
            fpr = compute_surrogate_fpr(
                series[src_name],
                series[tgt_name],
                n_surrogates=n_surrogates,
                rng_seed=seed,
            )
            results.setdefault(key, []).append(fpr)

        # D3 all-pairs
        for i, src_name in enumerate(_D3_PROCESS_VARS):
            for j, tgt_name in enumerate(_D3_PROCESS_VARS):
                if i == j:
                    continue
                key = f"D3:{src_name}->{tgt_name}"
                fpr = compute_surrogate_fpr(
                    series[src_name],
                    series[tgt_name],
                    n_surrogates=n_surrogates,
                    rng_seed=seed + 1000,
                )
                results.setdefault(key, []).append(fpr)

        log(f"  analyzed {regime}/seed_{seed:03d}")

    # Summarize
    summary: dict[str, dict] = {}
    for key, fprs in results.items():
        arr = np.array(fprs)
        summary[key] = {
            "mean_fpr": round(float(np.mean(arr)), 4),
            "std_fpr": round(float(np.std(arr)), 4),
            "max_fpr": round(float(np.max(arr)), 4),
            "n_seeds": len(fprs),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate FPR analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/benchmark"))
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--regime", default="E1")
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    data_dir = args.data_dir.resolve()

    log(f"Surrogate FPR: {args.regime}, seeds={args.seeds}, n_surrogates={args.n_surrogates}")

    summary = analyze_fpr(data_dir, seeds, args.regime, args.n_surrogates)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Written to {args.output}")
    else:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")

    # Report overall FPR
    all_means = [v["mean_fpr"] for v in summary.values()]
    if all_means:
        overall = np.mean(all_means)
        log(f"Overall mean FPR: {overall:.4f}")
        if overall > 0.10:
            log("WARNING: FPR exceeds 0.10 target")


if __name__ == "__main__":
    main()
