"""Shared adapter infrastructure: AdapterResult, time series extraction, utilities.

All adapters operate on a single RunSummary dict (Mode B) and produce
per-family scores by extracting family time series from family_breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# AdapterResult
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of scoring a single family against one definition."""

    definition: str  # "D1" | "D2" | "D3" | "D4"
    family_id: int  # 0=F1, 1=F2, 2=F3
    score: float  # S ∈ [0, 1], primary graded metric
    passes_threshold: bool  # binary label
    threshold_used: float
    criteria: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-family time series extraction
# ---------------------------------------------------------------------------

# Fields available in FamilyStepMetrics (per data contract)
_FAMILY_FIELDS = [
    "alive_count",
    "population_size",
    "energy_mean",
    "waste_mean",
    "boundary_mean",
    "birth_count",
    "death_count",
    "mean_generation",
    "mean_genome_drift",
    "genome_diversity",
    "maturity_mean",
]


def extract_family_series(run_summary: dict, family_id: int) -> dict[str, np.ndarray]:
    """Extract per-family time series from a Mode B RunSummary.

    Returns a dict mapping field name to numpy array of values across all
    sample steps, ordered chronologically.
    """
    samples = run_summary["samples"]
    series: dict[str, list[float]] = {f: [] for f in _FAMILY_FIELDS}

    for sample in samples:
        breakdown = sample.get("family_breakdown", [])
        fam = None
        for entry in breakdown:
            if entry["family_id"] == family_id:
                fam = entry
                break

        if fam is None:
            # Family not present in this step — append zeros
            for f in _FAMILY_FIELDS:
                series[f].append(0.0)
        else:
            for f in _FAMILY_FIELDS:
                series[f].append(float(fam.get(f, 0.0)))

    return {f: np.array(v, dtype=float) for f, v in series.items()}


def extract_family_lineage(run_summary: dict, family_id: int) -> list[dict]:
    """Extract lineage events for a specific family."""
    return [e for e in run_summary.get("lineage_events", []) if e.get("family_id") == family_id]


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid mapping x → [0, 1]. Used to map Cohen's d to a score."""
    return 1.0 / (1.0 + np.exp(-k * x))


def coefficient_of_variation(arr: np.ndarray) -> float:
    """Compute CV = std / |mean|. Returns 0 if mean is zero."""
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / abs(mean))


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction.

    Returns corrected p-values in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    # Work backwards to enforce monotonicity
    prev = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_from_end  # 1-based rank from smallest
        adjusted = p * n / rank
        adjusted = min(adjusted, prev)
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = adjusted
        prev = adjusted

    return corrected
