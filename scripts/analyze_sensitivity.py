"""Sensitivity analysis for D1 weights, binary thresholds, and D3 FDR q-value.

Three sweeps:
1. D1 weights: ±20% perturbation of (W_α, W_β, W_γ), report max Δ in mean score.
2. Binary threshold: sweep 0.1–0.9, report pass-rate monotonicity.
3. D3 FDR q: vary q ∈ {0.01, 0.05, 0.10}, report closure score changes.

Usage:
    uv run python scripts/analyze_sensitivity.py experiments/benchmark/ --seeds 0-99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from adapters.common import discover_family_ids
from adapters.d1 import W_ALPHA, W_BETA, W_GAMMA, score_d1
from adapters.d3 import score_d3

DEFINITIONS = ["D1", "D2", "D3", "D4"]

# D1 weight perturbation vectors: ±20% emphasis on each component, renormalized to sum=1.
def _norm(a: float, b: float, c: float) -> tuple[float, float, float]:
    s = a + b + c
    return (a / s, b / s, c / s)


D1_WEIGHT_PERTURBATIONS = [
    ("default", (W_ALPHA, W_BETA, W_GAMMA)),
    ("+20%α -20%β +20%γ", _norm(W_ALPHA * 1.2, W_BETA * 0.8, W_GAMMA * 1.2)),
    ("-20%α +20%β -20%γ", _norm(W_ALPHA * 0.8, W_BETA * 1.2, W_GAMMA * 0.8)),
    ("+20%α +20%β -20%γ", _norm(W_ALPHA * 1.2, W_BETA * 1.2, W_GAMMA * 0.8)),
    ("-20%α -20%β +20%γ", _norm(W_ALPHA * 0.8, W_BETA * 0.8, W_GAMMA * 1.2)),
]

D3_FDR_VALUES = [0.01, 0.05, 0.10]


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------


def sweep_d1_weights(data_dir: Path, seeds: list[int], regimes: list[str]) -> dict:
    """Sweep D1 weight perturbations across calibration data."""
    results = {}
    for label, weights in D1_WEIGHT_PERTURBATIONS:
        scores_by_family: dict[int, list[float]] = {}
        for regime in regimes:
            regime_dir = data_dir / regime
            for seed in seeds:
                path = regime_dir / f"seed_{seed:03d}.json"
                if not path.exists():
                    continue
                run = json.loads(path.read_text())
                for fid in discover_family_ids(run):
                    r = score_d1(run, family_id=fid, weights=weights)
                    scores_by_family.setdefault(fid, []).append(r.score)

        results[label] = {
            f"F{fid + 1}_mean": float(np.mean(scores))
            for fid, scores in sorted(scores_by_family.items())
        }

    return results


def sweep_thresholds(data_dir: Path, seeds: list[int], regimes: list[str]) -> dict:
    """Sweep binary threshold 0.1–0.9 and count passes per definition."""
    from adapters import score_all

    threshold_values = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    raw_scores: dict[str, list[float]] = {d: [] for d in DEFINITIONS}

    for regime in regimes:
        regime_dir = data_dir / regime
        for seed in seeds:
            path = regime_dir / f"seed_{seed:03d}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            for fid in discover_family_ids(run):
                results = score_all(run, family_id=fid)
                for defn in DEFINITIONS:
                    raw_scores[defn].append(results[defn].score)

    sweep_results = {}
    for defn in DEFINITIONS:
        scores = raw_scores[defn]
        if not scores:
            continue
        pass_rates = []
        for thresh in threshold_values:
            rate = float(np.mean([s >= thresh for s in scores]))
            pass_rates.append({"threshold": thresh, "pass_rate": rate})
        sweep_results[defn] = pass_rates

    return sweep_results


def sweep_d3_fdr(data_dir: Path, seeds: list[int], regimes: list[str]) -> dict:
    """Sweep D3 FDR q-value and report closure scores."""
    results = {}
    for q in D3_FDR_VALUES:
        closure_scores: list[float] = []
        edge_counts: list[int] = []
        for regime in regimes:
            regime_dir = data_dir / regime
            for seed in seeds:
                path = regime_dir / f"seed_{seed:03d}.json"
                if not path.exists():
                    continue
                run = json.loads(path.read_text())
                for fid in discover_family_ids(run):
                    r = score_d3(run, family_id=fid, fdr_q=q)
                    closure_scores.append(r.criteria["closure"])
                    edge_counts.append(r.metadata["n_significant_edges"])

        results[f"q={q}"] = {
            "mean_closure": float(np.mean(closure_scores)) if closure_scores else 0.0,
            "mean_edges": float(np.mean(edge_counts)) if edge_counts else 0.0,
            "n_runs": len(closure_scores),
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_seed_range(s: str) -> list[int]:
    """Parse '0-99' or '0,1,2' into a list of ints."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analysis sweeps")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--seeds", default="0-99", help="Seed range (e.g. 0-99)")
    parser.add_argument(
        "--regimes",
        default="E1,E2,E3,E4,E5",
        help="Comma-separated regime names",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    regimes = args.regimes.split(",")

    print(f"Running sensitivity sweeps on {args.data_dir}", file=sys.stderr)
    print(f"  Seeds: {seeds[0]}–{seeds[-1]} ({len(seeds)} seeds)", file=sys.stderr)
    print(f"  Regimes: {regimes}", file=sys.stderr)

    output = {
        "d1_weights": sweep_d1_weights(args.data_dir, seeds, regimes),
        "thresholds": sweep_thresholds(args.data_dir, seeds, regimes),
        "d3_fdr": sweep_d3_fdr(args.data_dir, seeds, regimes),
    }

    json_str = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(json_str)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
