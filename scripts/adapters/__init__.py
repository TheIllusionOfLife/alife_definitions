"""Adapter package for scoring runs against definitions D1–D4.

Entry point: ``score_all(run_summary, family_id)`` returns all four AdapterResults.
"""

from __future__ import annotations

from .common import AdapterResult


def score_all(
    run_summary: dict,
    family_id: int,
    *,
    thresholds: dict[str, float] | None = None,
    d3_mode: str = "closure_only",
    d1_weights: tuple[float, float, float] | None = None,
    d1_aggregation: str = "geometric",
    d3_fdr_q: float | None = None,
    d3_edge_mode: str = "bonferroni",
    d4_similarity_mode: str = "hash",
) -> dict[str, AdapterResult]:
    """Score a single family against all four definitions.

    Args:
        run_summary: Parsed JSON from run_experiment_json (Mode B).
        family_id: Family index (0=F1, 1=F2, 2=F3).
        thresholds: Optional per-definition thresholds {"D1": 0.5, ...}.
        d3_mode: D3 scoring mode ("closure_only" or "closure_x_persistence").
        d1_weights: Optional (w_alpha, w_beta, w_gamma) for D1 sensitivity.
        d1_aggregation: D1 aggregation mode (geometric/arithmetic/harmonic/min).
        d3_fdr_q: Optional FDR q-value for D3 sensitivity.
        d3_edge_mode: D3 edge combination mode ("bonferroni" or "and").
        d4_similarity_mode: D4 genome similarity mode ("hash" or "l2").

    Returns:
        Dict mapping definition name to AdapterResult.
    """
    from .d1 import score_d1
    from .d2 import score_d2
    from .d3 import score_d3
    from .d4 import score_d4

    thresholds = thresholds or {}
    results: dict[str, AdapterResult] = {}

    for name, fn in [("D1", score_d1), ("D2", score_d2), ("D3", score_d3), ("D4", score_d4)]:
        kwargs = {}
        if name in thresholds:
            kwargs["threshold"] = thresholds[name]
        if name == "D1":
            if d1_weights is not None:
                kwargs["weights"] = d1_weights
            kwargs["aggregation"] = d1_aggregation
        if name == "D3":
            kwargs["mode"] = d3_mode
            kwargs["edge_mode"] = d3_edge_mode
            if d3_fdr_q is not None:
                kwargs["fdr_q"] = d3_fdr_q
        if name == "D4":
            kwargs["similarity_mode"] = d4_similarity_mode
        results[name] = fn(run_summary, family_id=family_id, **kwargs)

    return results
