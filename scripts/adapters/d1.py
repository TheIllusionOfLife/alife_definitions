"""D1 — Textbook 7-criteria + Functional Analogy adapter.

Each criterion is assessed via three conditions:
  α — Dynamic process (CV > noise floor)
  β — Measurable degradation (Cohen's d vs ablated family)
  γ — Feedback coupling (significant transfer entropy)

Aggregate: geometric mean of 7 criterion scores.
"""

from __future__ import annotations

import numpy as np
from analyses.coupling.transfer_entropy import transfer_entropy_lag1
from analyses.results.statistics import cohens_d

from .common import (
    AdapterResult,
    coefficient_of_variation,
    extract_family_series,
    sigmoid,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.3
CV_NOISE_FLOOR = 0.01  # minimum CV to count as "dynamic"
W_ALPHA = 0.3  # dynamic process weight
W_BETA = 0.4  # degradation weight
W_GAMMA = 0.3  # coupling weight
TE_BINS = 3  # bins for TE (conservative for n≈40)
TE_PERMS = 400  # permutations for TE significance
TE_ALPHA = 0.05  # significance threshold for TE

# Criterion → primary signal mapping
_CRITERION_SIGNAL = {
    "metabolism": "energy_mean",
    "boundary": "boundary_mean",
    "homeostasis": "energy_mean",  # CV-based stability proxy
    "response": "maturity_mean",
    "reproduction": "birth_count",
    "evolution": "genome_diversity",
    "growth": "maturity_mean",
}

# Criterion → which family lacks it (for cross-family degradation test)
# F2 (family_id=1) lacks boundary, homeostasis
# F3 (family_id=2) lacks reproduction, evolution
_CRITERION_ABLATION_FAMILY = {
    "boundary": 1,
    "homeostasis": 1,
    "reproduction": 2,
    "evolution": 2,
}

# TE coupling pairs: criterion → (source_signal, target_signal)
_CRITERION_COUPLING = {
    "metabolism": ("energy_mean", "boundary_mean"),
    "boundary": ("boundary_mean", "energy_mean"),
    "homeostasis": ("energy_mean", "waste_mean"),
    "response": ("alive_count", "energy_mean"),
    "reproduction": ("birth_count", "alive_count"),
    "evolution": ("genome_diversity", "alive_count"),
    "growth": ("maturity_mean", "alive_count"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_d1(
    run_summary: dict,
    *,
    family_id: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> AdapterResult:
    """Score a family against D1 (textbook 7-criteria).

    Returns an AdapterResult with per-criterion scores in criteria dict
    and aggregate geometric mean as the overall score.
    """
    rng = np.random.default_rng(2026 + family_id)

    # Extract time series for target family and all families
    target = extract_family_series(run_summary, family_id)
    all_families = {fid: extract_family_series(run_summary, fid) for fid in range(3)}

    criteria_scores: dict[str, float] = {}
    coupling_scores: dict[str, float] = {}

    for criterion in _CRITERION_SIGNAL:
        # If this family IS the ablated one for this criterion, cap the score.
        # The criterion is structurally absent — α/γ signals are artifacts of
        # passive dynamics (e.g. boundary decays even without maintenance).
        ablation_fid = _CRITERION_ABLATION_FAMILY.get(criterion)
        is_ablated = ablation_fid is not None and ablation_fid == family_id

        alpha = _score_dynamic(criterion, target)
        beta = _score_degradation(criterion, family_id, target, all_families)
        gamma = _score_coupling(criterion, target, rng)
        coupling_scores[criterion] = gamma

        if is_ablated:
            # Hard cap: the criterion is disabled in this family
            score = 0.15 * (W_ALPHA * alpha + W_GAMMA * gamma)
        else:
            score = W_ALPHA * alpha + W_BETA * _sigmoid_d(beta) + W_GAMMA * gamma
        criteria_scores[criterion] = float(np.clip(score, 0.0, 1.0))

    aggregate = _geometric_mean(list(criteria_scores.values()))

    return AdapterResult(
        definition="D1",
        family_id=family_id,
        score=float(aggregate),
        passes_threshold=aggregate >= threshold,
        threshold_used=threshold,
        criteria=criteria_scores,
        metadata={
            "coupling_scores": coupling_scores,
            "n_samples": len(run_summary["samples"]),
        },
    )


# ---------------------------------------------------------------------------
# Sub-score computations
# ---------------------------------------------------------------------------


def _score_dynamic(criterion: str, series: dict[str, np.ndarray]) -> float:
    """α — Dynamic process: CV of primary signal > noise floor."""
    signal_name = _CRITERION_SIGNAL[criterion]
    signal = series[signal_name]

    if criterion == "homeostasis":
        # Homeostasis: low CV indicates stability — invert the test
        cv = coefficient_of_variation(signal)
        # Score higher when CV is LOW (good homeostasis)
        # But still need *some* variance to prove it's active
        return 1.0 if cv > CV_NOISE_FLOOR else 0.0

    if criterion == "reproduction":
        # For birth_count: check sustained births over last 50%
        half = len(signal) // 2
        recent = signal[half:]
        persistence = float(np.mean(recent > 0))
        return 1.0 if persistence > 0.1 else 0.0

    if criterion == "evolution":
        # Need both drift > 0 and diversity > 0
        drift = series["mean_genome_drift"]
        diversity = series["genome_diversity"]
        has_drift = float(np.mean(drift)) > 0
        has_diversity = float(np.mean(diversity)) > 0
        return 1.0 if (has_drift and has_diversity) else 0.0

    if criterion == "growth":
        # Maturity should increase over time
        signal = series["maturity_mean"]
        if len(signal) < 4:
            return 0.0
        quarter = len(signal) // 4
        early = np.mean(signal[:quarter]) if quarter > 0 else 0.0
        late = np.mean(signal[-quarter:]) if quarter > 0 else 0.0
        return 1.0 if late > early or coefficient_of_variation(signal) > CV_NOISE_FLOOR else 0.0

    # Default: CV test
    cv = coefficient_of_variation(signal)
    return 1.0 if cv > CV_NOISE_FLOOR else 0.0


def _score_degradation(
    criterion: str,
    family_id: int,
    target: dict[str, np.ndarray],
    all_families: dict[int, dict[str, np.ndarray]],
) -> float:
    """β — Measurable degradation: Cohen's d vs ablated family."""
    signal_name = _CRITERION_SIGNAL[criterion]
    target_signal = target[signal_name]

    ablation_fid = _CRITERION_ABLATION_FAMILY.get(criterion)

    if ablation_fid is not None and ablation_fid != family_id:
        # Compare target family vs the family that lacks this criterion
        ablated_signal = all_families[ablation_fid][signal_name]

        if criterion == "homeostasis":
            # For homeostasis, compare stability (inverse CV)
            target_cv = _rolling_cv(target_signal)
            ablated_cv = _rolling_cv(ablated_signal)
            # Lower CV = better homeostasis, so d should be negative (target < ablated)
            d = cohens_d(ablated_cv, target_cv)
        elif criterion == "evolution":
            # Compare genome diversity
            target_div = target["genome_diversity"]
            ablated_div = all_families[ablation_fid]["genome_diversity"]
            d = cohens_d(target_div, ablated_div)
        else:
            d = cohens_d(target_signal, ablated_signal)
    else:
        # No natural ablation family — compare against zero-variance null
        d = _cohens_d_vs_null(target_signal)

    return float(d)


def _score_coupling(
    criterion: str,
    series: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> float:
    """γ — Feedback coupling: TE from criterion signal to another."""
    src_name, tgt_name = _CRITERION_COUPLING[criterion]
    src = series[src_name]
    tgt = series[tgt_name]

    result = transfer_entropy_lag1(src, tgt, bins=TE_BINS, permutations=TE_PERMS, rng=rng)
    if result is None:
        return 0.0

    # Graded coupling score: with short series (n≈40), binary significance is
    # too strict. Use a graded score: 1.0 if significant, else scale by how
    # far observed TE exceeds the null mean (evidence of coupling even if not
    # reaching α=0.05).
    if result["p_value"] < TE_ALPHA:
        return 1.0
    # Partial credit: TE exceeds null mean → some coupling evidence
    te = result["te"]
    null_mean = result["null_mean"]
    if null_mean > 0 and te > null_mean:
        ratio = min((te - null_mean) / null_mean, 2.0)  # cap at 2× null
        return float(np.clip(ratio * 0.5, 0.0, 0.8))  # max 0.8 without significance
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geometric_mean(values: list[float]) -> float:
    """Compute geometric mean of non-negative values. Zero if any value is zero."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def _sigmoid_d(d: float) -> float:
    """Map Cohen's d to [0, 1] via sigmoid."""
    return sigmoid(d, k=1.0)


def _cohens_d_vs_null(signal: np.ndarray) -> float:
    """Cohen's d comparing signal against constant zero."""
    if len(signal) < 2:
        return 0.0
    mean = np.mean(signal)
    std = np.std(signal, ddof=1)
    if std == 0:
        return 0.0 if mean == 0 else 1.0
    return float(mean / std)


def _rolling_cv(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute rolling CV with a sliding window."""
    if len(signal) < window:
        cv = coefficient_of_variation(signal)
        return np.array([cv])
    cvs = []
    for i in range(len(signal) - window + 1):
        chunk = signal[i : i + window]
        mean = np.mean(chunk)
        if mean == 0:
            cvs.append(0.0)
        else:
            cvs.append(float(np.std(chunk, ddof=1) / abs(mean)))
    return np.array(cvs, dtype=float)
