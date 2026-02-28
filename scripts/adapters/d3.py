"""D3 — Autonomy / Organizational Closure adapter.

Operationalizes Varela/Maturana autopoiesis as a measurable graph property:
the largest strongly connected component (SCC) among process variables
represents organizational closure.

Process variables: energy, waste, boundary, birth_count, maturity.
Edges detected via transfer entropy + Granger causality.
"""

from __future__ import annotations

import numpy as np
from analyses.coupling.granger import best_granger_with_lag_correction
from analyses.coupling.transfer_entropy import transfer_entropy_lag1

from .common import (
    AdapterResult,
    benjamini_hochberg,
    extract_family_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.3
TE_BINS = 5  # 5 bins viable for n≈200 with xcorr complement
TE_PERMS = 400
GRANGER_MAX_LAG = 5
FDR_Q = 0.05

# Process variables for closure analysis
PROCESS_VARS = ["energy_mean", "waste_mean", "boundary_mean", "birth_count", "maturity_mean"]
N_PROCESSES = len(PROCESS_VARS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_d3(
    run_summary: dict,
    *,
    family_id: int,
    threshold: float = DEFAULT_THRESHOLD,
    mode: str = "closure_only",
    fdr_q: float | None = None,
) -> AdapterResult:
    """Score a family against D3 (autonomy/organizational closure).

    Args:
        mode: ``"closure_only"`` (default) uses pure closure as the score,
              addressing reviewer concern about circularity with alive-count AUC.
              ``"closure_x_persistence"`` uses closure × persistence (legacy).
              Both values are always available in ``criteria``.
        fdr_q: Optional FDR q-value override for sensitivity analysis.
               Defaults to module-level ``FDR_Q`` (0.05).
    """
    if mode not in ("closure_only", "closure_x_persistence"):
        raise ValueError(f"Invalid D3 mode: {mode!r}")

    q = fdr_q if fdr_q is not None else FDR_Q

    rng = np.random.default_rng(3026 + family_id)
    series = extract_family_series(run_summary, family_id)

    # Build directed influence matrix
    edges, n_significant, edge_details = _build_influence_graph(series, rng, q=q)

    # Find largest SCC — singleton SCCs don't count as closure
    scc_size = _largest_scc_size(edges)
    closure = 0.0 if scc_size <= 1 else scc_size / N_PROCESSES

    # Persistence requirement — measures self-maintenance duration.
    alive = series["alive_count"]
    if len(alive) >= 2 and alive[0] > 0:
        survival_frac = float(np.mean(alive > alive[0] * 0.5))
        endpoint_ratio = float(np.clip(alive[-1] / alive[0], 0.0, 1.0))
        persistence = max(survival_frac, endpoint_ratio)
    else:
        persistence = 0.0

    # Both score variants always computed
    score_closure_only = closure
    score_closure_x_persistence = closure * persistence

    # Primary score follows mode
    score = score_closure_only if mode == "closure_only" else score_closure_x_persistence

    return AdapterResult(
        definition="D3",
        family_id=family_id,
        score=float(np.clip(score, 0.0, 1.0)),
        passes_threshold=score >= threshold,
        threshold_used=threshold,
        criteria={
            "closure": float(closure),
            "persistence": float(persistence),
            "score_closure_only": float(np.clip(score_closure_only, 0.0, 1.0)),
            "score_closure_x_persistence": float(np.clip(score_closure_x_persistence, 0.0, 1.0)),
        },
        metadata={
            "largest_scc_size": scc_size,
            "n_significant_edges": n_significant,
            "n_possible_edges": N_PROCESSES * (N_PROCESSES - 1),
            "n_samples": len(run_summary["samples"]),
            "mode": mode,
        },
    )


# ---------------------------------------------------------------------------
# Influence graph construction
# ---------------------------------------------------------------------------


def _build_influence_graph(
    series: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    q: float = FDR_Q,
) -> tuple[list[tuple[int, int]], int, list[dict]]:
    """Build directed influence graph from TE and Granger tests.

    Returns:
        edges: List of (i, j) directed edges where i→j is significant.
        n_significant: Number of significant edges after FDR correction.
        details: Per-pair test results for debugging.
    """
    # Collect all p-values for FDR correction
    pair_results: list[dict] = []

    for i, src_name in enumerate(PROCESS_VARS):
        for j, tgt_name in enumerate(PROCESS_VARS):
            if i == j:
                continue
            src = series[src_name]
            tgt = series[tgt_name]

            te_result = transfer_entropy_lag1(
                src, tgt, bins=TE_BINS, permutations=TE_PERMS, rng=rng
            )
            te_p = te_result["p_value"] if te_result else 1.0

            granger_result = best_granger_with_lag_correction(src, tgt, GRANGER_MAX_LAG)
            granger_p = granger_result["best_p_corrected"] if granger_result else 1.0

            # Combine two tests conservatively per pair before global FDR.
            # Bonferroni for m=2: p_pair = min(1, 2 * min(p1, p2))
            min_p = min(1.0, 2.0 * min(te_p, granger_p))

            pair_results.append(
                {
                    "src": src_name,
                    "tgt": tgt_name,
                    "src_idx": i,
                    "tgt_idx": j,
                    "te_p": te_p,
                    "granger_p": granger_p,
                    "min_p": min_p,
                }
            )

    # BH-FDR correction across all pairs
    raw_ps = [r["min_p"] for r in pair_results]
    corrected_ps = benjamini_hochberg(raw_ps, q=q)

    edges: list[tuple[int, int]] = []
    n_significant = 0
    for r, p_corr in zip(pair_results, corrected_ps, strict=True):
        r["p_corrected"] = p_corr
        r["significant"] = p_corr <= q
        if r["significant"]:
            edges.append((r["src_idx"], r["tgt_idx"]))
            n_significant += 1

    return edges, n_significant, pair_results


# ---------------------------------------------------------------------------
# SCC computation (Tarjan's algorithm)
# ---------------------------------------------------------------------------


def _largest_scc_size(edges: list[tuple[int, int]]) -> int:
    """Find the size of the largest strongly connected component.

    Uses iterative Tarjan's algorithm on a small graph (5 nodes).
    """
    # Build adjacency list
    adj: dict[int, list[int]] = {i: [] for i in range(N_PROCESSES)}
    for src, tgt in edges:
        adj[src].append(tgt)

    # Tarjan's iterative
    index_counter = [0]
    stack: list[int] = []
    on_stack = [False] * N_PROCESSES
    index = [-1] * N_PROCESSES
    lowlink = [-1] * N_PROCESSES
    sccs: list[list[int]] = []

    def strongconnect(v: int) -> None:
        work_stack: list[tuple[int, int]] = [(v, 0)]
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        while work_stack:
            node, ni = work_stack[-1]
            if ni < len(adj[node]):
                work_stack[-1] = (node, ni + 1)
                w = adj[node][ni]
                if index[w] == -1:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    work_stack.append((w, 0))
                elif on_stack[w]:
                    lowlink[node] = min(lowlink[node], index[w])
            else:
                if lowlink[node] == index[node]:
                    scc: list[int] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    sccs.append(scc)
                work_stack.pop()
                if work_stack:
                    parent = work_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for v in range(N_PROCESSES):
        if index[v] == -1:
            strongconnect(v)

    if not sccs:
        return 0
    return max(len(scc) for scc in sccs)
