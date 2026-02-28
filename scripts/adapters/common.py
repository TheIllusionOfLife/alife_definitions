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


def discover_family_ids(run_summary: dict) -> list[int]:
    """Discover all family IDs present in the run data.

    Scans the first sample's family_breakdown to find all family IDs,
    rather than assuming a fixed set.
    """
    samples = run_summary.get("samples", [])
    if not samples:
        return []
    breakdown = samples[0].get("family_breakdown", [])
    return sorted(entry["family_id"] for entry in breakdown)


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


def lagged_cross_correlation_score(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
) -> float:
    """Compute a coupling score from lagged Pearson correlations.

    Returns the maximum absolute Pearson r across lags 0..max_lag,
    mapped to [0, 1]. Short series (< max_lag + 3) return 0.0.
    """
    if len(x) < max_lag + 3 or len(y) < max_lag + 3:
        return 0.0
    best_r = 0.0
    for lag in range(max_lag + 1):
        if lag == 0:
            x_s, y_s = x, y
        else:
            x_s, y_s = x[:-lag], y[lag:]
        if len(x_s) < 3:
            continue
        if np.std(x_s) == 0 or np.std(y_s) == 0:
            continue
        r = float(np.abs(np.corrcoef(x_s, y_s)[0, 1]))
        if not np.isnan(r) and r > best_r:
            best_r = r
    return float(np.clip(best_r, 0.0, 1.0))


def compute_surrogate_fpr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_surrogates: int = 100,
    rng_seed: int = 42,
    bins: int = 3,
    permutations: int = 200,
    alpha: float = 0.05,
) -> float:
    """Compute false positive rate using phase-randomized surrogates.

    Generates n_surrogates surrogate pairs via phase randomization,
    runs TE on each, and returns the fraction that pass significance.
    """
    from analyses.coupling.transfer_entropy import phase_randomize, transfer_entropy_lag1

    rng = np.random.default_rng(rng_seed)
    n_significant = 0
    n_valid = 0
    for _ in range(n_surrogates):
        x_surr = phase_randomize(x, rng)
        y_surr = phase_randomize(y, rng)
        result = transfer_entropy_lag1(
            x_surr, y_surr, bins=bins, permutations=permutations, rng=rng
        )
        if result is not None:
            n_valid += 1
            if result["p_value"] < alpha:
                n_significant += 1
    if n_valid == 0:
        return 0.0
    return n_significant / n_valid


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
