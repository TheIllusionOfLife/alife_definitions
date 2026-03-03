"""Temporal stability analysis of D3 organizational closure (SCC membership).

Splits runs into time windows and measures Jaccard similarity of SCC
membership between windows to assess whether closure is stable or transient.

Usage:
    uv run python scripts/analyze_temporal_d3.py experiments/benchmark/ \
        --seeds 0-99 --regimes E1,E2,E3,E4,E5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from adapters.common import (
    benjamini_hochberg,
    discover_family_ids,
    extract_family_series,
    find_all_sccs,
)
from adapters.d3 import (
    FDR_Q,
    GRANGER_MAX_LAG,
    N_PROCESSES,
    PROCESS_VARS,
    TE_BINS,
    TE_PERMS,
)
from analyses.coupling.granger import best_granger_with_lag_correction
from analyses.coupling.transfer_entropy import transfer_entropy_lag1
from experiment_common import log, parse_seed_range, safe_path


def _build_windowed_graph(
    series: dict[str, np.ndarray],
    start: int,
    end: int,
    rng: np.random.Generator,
) -> set[int]:
    """Build influence graph on a time window and return SCC member indices."""
    # Slice series to window
    windowed = {k: v[start:end] for k, v in series.items()}

    pair_results: list[dict] = []
    for i, src_name in enumerate(PROCESS_VARS):
        for j, tgt_name in enumerate(PROCESS_VARS):
            if i == j:
                continue
            src = windowed[src_name]
            tgt = windowed[tgt_name]

            te_result = transfer_entropy_lag1(
                src, tgt, bins=TE_BINS, permutations=TE_PERMS, rng=rng
            )
            te_p = te_result["p_value"] if te_result else 1.0

            granger_result = best_granger_with_lag_correction(src, tgt, GRANGER_MAX_LAG)
            granger_p = granger_result["best_p_corrected"] if granger_result else 1.0

            combined_p = min(1.0, 2.0 * min(te_p, granger_p))
            pair_results.append({"src_idx": i, "tgt_idx": j, "combined_p": combined_p})

    raw_ps = [r["combined_p"] for r in pair_results]
    corrected_ps = benjamini_hochberg(raw_ps, q=FDR_Q)

    edges: list[tuple[int, int]] = []
    for r, p_corr in zip(pair_results, corrected_ps, strict=True):
        if p_corr <= FDR_Q:
            edges.append((r["src_idx"], r["tgt_idx"]))

    # Find SCC members using shared utility
    sccs = find_all_sccs(edges, N_PROCESSES)
    if not sccs or len(sccs[0]) <= 1:
        return set()
    return set(sccs[0])


def analyze_temporal_d3(
    data_dir: Path,
    seeds: list[int],
    regimes: list[str],
    n_windows: int = 2,
) -> dict:
    """Analyze temporal stability of D3 SCC membership across time windows."""
    jaccard_scores: list[float] = []
    window_closures: dict[int, list[float]] = {w: [] for w in range(n_windows)}

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
                n_samples = len(series["alive_count"])
                if n_samples < n_windows * 10:
                    continue

                window_size = n_samples // n_windows
                window_sccs: list[set[int]] = []

                for w in range(n_windows):
                    start = w * window_size
                    end = start + window_size if w < n_windows - 1 else n_samples
                    rng = np.random.default_rng(3026 + fid + w * 1000 + seed)
                    scc = _build_windowed_graph(series, start, end, rng)
                    window_sccs.append(scc)
                    closure = len(scc) / N_PROCESSES if len(scc) > 1 else 0.0
                    window_closures[w].append(closure)

                # Compute pairwise Jaccard between consecutive windows
                for w in range(len(window_sccs) - 1):
                    s1 = window_sccs[w]
                    s2 = window_sccs[w + 1]
                    if not s1 and not s2:
                        jaccard = 1.0  # Both empty → perfect agreement
                    elif not s1 or not s2:
                        jaccard = 0.0
                    else:
                        jaccard = len(s1 & s2) / len(s1 | s2)
                    jaccard_scores.append(jaccard)

            log(f"  {regime}/seed_{seed:03d}")

    per_window = {}
    for w, closures in window_closures.items():
        per_window[f"window_{w}"] = {
            "mean_closure": round(float(np.mean(closures)), 4) if closures else 0.0,
            "std_closure": round(float(np.std(closures)), 4) if closures else 0.0,
            "n_runs": len(closures),
        }

    return {
        "n_windows": n_windows,
        "mean_jaccard": round(float(np.mean(jaccard_scores)), 4) if jaccard_scores else 0.0,
        "std_jaccard": round(float(np.std(jaccard_scores)), 4) if jaccard_scores else 0.0,
        "n_comparisons": len(jaccard_scores),
        "per_window": per_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal D3 SCC stability analysis")
    parser.add_argument("data_dir", type=Path, help="Benchmark data directory")
    parser.add_argument("--seeds", default="0-99", help="Seed range")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5")
    parser.add_argument("--n-windows", type=int, default=2, help="Number of time windows")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    regimes = [r.strip() for r in args.regimes.split(",")]

    log(f"Temporal D3: {args.n_windows} windows, seeds={args.seeds}")

    result = analyze_temporal_d3(args.data_dir.resolve(), seeds, regimes, args.n_windows)

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log(f"Written to {args.output}")
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
