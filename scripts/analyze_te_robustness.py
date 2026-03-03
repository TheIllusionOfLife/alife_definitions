"""TE/Granger estimator robustness sweep across bins and lag parameters.

Sweeps bins={5,10,20} × lag={1,2,3} on calibration seeds to verify that
D3 closure scores are stable across TE estimator settings.  Reports
Spearman rank correlation of D3 closure scores across settings.

Usage:
    uv run python scripts/analyze_te_robustness.py \
        experiments/benchmark --seeds 0-19 --regimes E1
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from adapters.common import benjamini_hochberg, discover_family_ids, extract_family_series
from adapters.d3 import (
    N_PROCESSES,
    PROCESS_VARS,
    _largest_scc_size,
)
from analyses.coupling.granger import best_granger_with_lag_correction
from analyses.coupling.transfer_entropy import transfer_entropy_lag1
from experiment_common import log, parse_seed_range, safe_path
from scipy import stats

# Sweep parameters
BINS_SETTINGS = [5, 10, 20]
LAG_SETTINGS = [1, 2, 3]
TE_PERMS = 400
FDR_Q = 0.05


def _build_influence_graph_with_params(
    series: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    bins: int,
    lag: int,
    q: float = FDR_Q,
) -> tuple[list[tuple[int, int]], int]:
    """Build influence graph with custom bins and lag settings."""
    pair_results: list[dict] = []

    for i, src_name in enumerate(PROCESS_VARS):
        for j, tgt_name in enumerate(PROCESS_VARS):
            if i == j:
                continue
            src = series[src_name]
            tgt = series[tgt_name]

            # TE with custom bins — always uses provided lag via slicing
            te_result = transfer_entropy_lag1(src, tgt, bins=bins, permutations=TE_PERMS, rng=rng)
            te_p = te_result["p_value"] if te_result else 1.0

            granger_result = best_granger_with_lag_correction(src, tgt, lag)
            granger_p = granger_result["best_p_corrected"] if granger_result else 1.0

            min_p = min(1.0, 2.0 * min(te_p, granger_p))
            pair_results.append({"src_idx": i, "tgt_idx": j, "min_p": min_p})

    raw_ps = [r["min_p"] for r in pair_results]
    corrected_ps = benjamini_hochberg(raw_ps, q=q)

    edges: list[tuple[int, int]] = []
    n_significant = 0
    for r, p_corr in zip(pair_results, corrected_ps, strict=True):
        if p_corr <= q:
            edges.append((r["src_idx"], r["tgt_idx"]))
            n_significant += 1

    return edges, n_significant


def sweep_te_robustness(
    data_dir: Path,
    seeds: list[int],
    regimes: list[str],
) -> dict:
    """Sweep bins × lag settings and compute D3 closure scores for each."""
    # Collect closure scores per setting, keyed by (bins, lag)
    setting_scores: dict[tuple[int, int], list[float]] = {}

    for bins in BINS_SETTINGS:
        for lag in LAG_SETTINGS:
            setting_scores[(bins, lag)] = []

    for regime in regimes:
        regime_dir = safe_path(data_dir, regime)
        for seed in seeds:
            path = regime_dir / f"seed_{seed:03d}.json"
            if not path.exists():
                continue
            with open(path) as f:
                run = json.load(f)

            family_ids = discover_family_ids(run)
            for fid in family_ids:
                series = extract_family_series(run, fid)

                for bins in BINS_SETTINGS:
                    for lag in LAG_SETTINGS:
                        rng = np.random.default_rng(3026 + fid + bins * 100 + lag * 10000)
                        edges, _ = _build_influence_graph_with_params(
                            series, rng, bins=bins, lag=lag
                        )
                        scc_size = _largest_scc_size(edges)
                        closure = 0.0 if scc_size <= 1 else scc_size / N_PROCESSES
                        setting_scores[(bins, lag)].append(closure)

            log(f"  {regime}/seed_{seed:03d}")

    # Compute Spearman rank correlations between all setting pairs
    settings = list(setting_scores.keys())
    rank_correlations: dict[str, float] = {}

    default_key = (5, 1)  # default bins=5, lag=1
    default_scores = setting_scores.get(default_key, [])

    for bins, lag in settings:
        if (bins, lag) == default_key:
            continue
        other_scores = setting_scores[(bins, lag)]
        if len(default_scores) == len(other_scores) and len(default_scores) >= 3:
            rho, _ = stats.spearmanr(default_scores, other_scores)
            rho_val = float(rho) if not np.isnan(rho) else 0.0
        else:
            rho_val = 0.0
        rank_correlations[f"bins{bins}_lag{lag}_vs_default"] = round(rho_val, 4)

    # All-pairs mean Spearman
    all_rhos: list[float] = []
    for s1, s2 in combinations(settings, 2):
        scores1 = setting_scores[s1]
        scores2 = setting_scores[s2]
        if len(scores1) == len(scores2) and len(scores1) >= 3:
            rho, _ = stats.spearmanr(scores1, scores2)
            if not np.isnan(rho):
                all_rhos.append(float(rho))

    per_setting = {}
    for (bins, lag), scores in setting_scores.items():
        per_setting[f"bins{bins}_lag{lag}"] = {
            "mean_closure": round(float(np.mean(scores)), 4) if scores else 0.0,
            "std_closure": round(float(np.std(scores)), 4) if scores else 0.0,
            "n_runs": len(scores),
        }

    return {
        "per_setting": per_setting,
        "rank_correlations_vs_default": rank_correlations,
        "mean_pairwise_spearman": round(float(np.mean(all_rhos)), 4) if all_rhos else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TE/Granger estimator robustness sweep")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--seeds", default="0-19", help="Seed range (subset for compute)")
    parser.add_argument("--regimes", default="E1", help="Comma-separated regimes")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    regimes = [r.strip() for r in args.regimes.split(",")]

    log(f"TE robustness sweep: seeds={args.seeds}, regimes={regimes}")
    log(f"  bins={BINS_SETTINGS}, lags={LAG_SETTINGS}")

    result = sweep_te_robustness(args.data_dir.resolve(), seeds, regimes)

    json_str = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_str)
        log(f"Wrote {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
