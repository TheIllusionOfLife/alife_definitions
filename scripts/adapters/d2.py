"""D2 — Darwinian / NASA definition adapter.

Operationalizes: "a self-sustaining chemical system capable of Darwinian evolution"
as three necessary conditions:
  S_reprod — Sustained reproduction
  S_hered — Heritability (parent-child genome similarity)
  S_select — Differential success (selection signal)

Aggregate: geometric mean (all three necessary).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .common import (
    AdapterResult,
    extract_family_lineage,
    extract_family_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.3
GENOME_LENGTH = 256  # matches analyze_evolution_evidence.py


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_d2(
    run_summary: dict,
    *,
    family_id: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> AdapterResult:
    """Score a family against D2 (Darwinian/NASA definition)."""
    series = extract_family_series(run_summary, family_id)
    lineage = extract_family_lineage(run_summary, family_id)

    s_reprod = _score_reproduction(series, lineage)
    s_hered = _score_heritability(lineage)
    s_select = _score_selection(series)

    criteria = {
        "S_reprod": s_reprod,
        "S_hered": s_hered,
        "S_select": s_select,
    }

    # Geometric mean — all three necessary
    components = [s_reprod, s_hered, s_select]
    if any(c <= 0 for c in components):
        aggregate = 0.0
    else:
        aggregate = float(np.exp(np.mean(np.log(components))))

    return AdapterResult(
        definition="D2",
        family_id=family_id,
        score=float(np.clip(aggregate, 0.0, 1.0)),
        passes_threshold=aggregate >= threshold,
        threshold_used=threshold,
        criteria=criteria,
        metadata={
            "n_lineage_events": len(lineage),
            "n_samples": len(run_summary["samples"]),
        },
    )


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------


def _score_reproduction(
    series: dict[str, np.ndarray],
    lineage: list[dict],
) -> float:
    """S_reprod — Sustained reproduction over last 50% of run.

    Uses both per-step birth_count from family_breakdown AND lineage events
    as complementary evidence. With sample_every=50, sporadic births may
    not appear in every window but lineage events capture them.
    """
    births = series["birth_count"]

    if len(births) == 0 and not lineage:
        return 0.0

    # Primary: persistence of per-step births in last 50%
    persistence = 0.0
    if len(births) > 0:
        half = len(births) // 2
        recent = births[half:]
        if len(recent) > 0:
            persistence = float(np.mean(recent > 0))

    # Secondary: lineage event rate (births per step)
    # A family with many lineage events but sparse per-step counts still reproduces
    lineage_score = 0.0
    if lineage:
        n_events = len(lineage)
        # Scale: 50+ events is strong sustained reproduction
        lineage_score = float(np.clip(n_events / 50.0, 0.0, 1.0))

    # Combine: take the max of both signals
    combined = max(persistence / 0.8, lineage_score)
    return float(np.clip(combined, 0.0, 1.0))


def _score_heritability(lineage: list[dict]) -> float:
    """S_hered — Heritability from genome hash similarity across generations.

    Uses analytical h² = 1 - V_mutation / (V_mutation + V_standing).
    Falls back to genome hash diversity if insufficient lineage data.
    """
    if not lineage:
        return 0.0

    # Group genome hashes by generation
    gen_hashes: dict[int, list[int]] = {}
    for event in lineage:
        gen = event.get("generation", 0)
        genome_hash = event.get("genome_hash", 0)
        gen_hashes.setdefault(gen, []).append(genome_hash)

    if len(gen_hashes) < 2:
        # Only one generation observed — can't measure cross-generation
        # But lineage events exist, so some heritability is present
        return 0.3

    # Compute intergenerational hash stability
    # Compare hash diversity within each generation vs across generations
    all_hashes = [h for events in gen_hashes.values() for h in events]
    unique_total = len(set(all_hashes))
    total = len(all_hashes)

    if total <= 1:
        return 0.0

    # Hash diversity ratio: low unique/total means high similarity (heritability)
    diversity_ratio = unique_total / total

    # Analytical h² approximation
    # Use mutation parameters from tuned baseline
    point_rate = 0.02  # from tuned_baseline
    mutation_scale = 0.15
    mutation_variance = point_rate * GENOME_LENGTH * mutation_scale**2

    # Standing variance proxied by hash diversity
    standing_variance = diversity_ratio * 100  # scale to comparable range

    h2 = 1.0 - mutation_variance / (mutation_variance + standing_variance)
    return float(np.clip(h2, 0.0, 1.0))


def _score_selection(series: dict[str, np.ndarray]) -> float:
    """S_select — Differential success via genome-fitness correlation."""
    genome_div = series["genome_diversity"]
    alive = series["alive_count"]

    if len(genome_div) < 4 or np.std(genome_div) == 0 or np.std(alive) == 0:
        return 0.0

    # Spearman correlation between genome diversity and alive count
    rho, _ = stats.spearmanr(genome_div, alive)
    if np.isnan(rho):
        return 0.0

    # Also check for directional drift
    drift = series["mean_genome_drift"]
    half = len(drift) // 2
    if half > 0:
        early = drift[:half]
        late = drift[half:]
        if len(early) >= 2 and len(late) >= 2 and np.std(early) > 0:
            _, p_drift = stats.mannwhitneyu(early, late, alternative="two-sided")
            drift_bonus = 0.2 if p_drift < 0.05 else 0.0
        else:
            drift_bonus = 0.0
    else:
        drift_bonus = 0.0

    return float(np.clip(abs(rho) + drift_bonus, 0.0, 1.0))
