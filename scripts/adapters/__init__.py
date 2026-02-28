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
) -> dict[str, AdapterResult]:
    """Score a single family against all four definitions.

    Args:
        run_summary: Parsed JSON from run_experiment_json (Mode B).
        family_id: Family index (0=F1, 1=F2, 2=F3).
        thresholds: Optional per-definition thresholds {"D1": 0.5, ...}.
        d3_mode: D3 scoring mode ("closure_only" or "closure_x_persistence").

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
        if name == "D3":
            kwargs["mode"] = d3_mode
        results[name] = fn(run_summary, family_id=family_id, **kwargs)

    return results
