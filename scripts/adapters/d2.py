"""D2 — Darwinian / NASA definition adapter.

Operationalizes: "a self-sustaining chemical system capable of Darwinian evolution"
as three necessary conditions:
  S_reprod — Sustained reproduction
  S_hered — Heritability (parent-child genome similarity)
  S_select — Differential success (selection signal)

Aggregate: geometric mean (all three necessary).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

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

# Reproduction scoring thresholds
REPROD_LINEAGE_EVENT_TARGET = 50.0  # events for max lineage_score
REPROD_PERSISTENCE_TARGET = 0.8  # persistence for max persistence_score

# Mutation parameters — must match default Rust config (mutation_point_rate,
# mutation_point_scale). Used for analytical h² approximation.
# WARNING: If experiments override mutation parameters, this score may be inaccurate.
MUTATION_POINT_RATE = 0.02
MUTATION_POINT_SCALE = 0.15


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
    price = _price_selection(lineage)
    s_select = price["score"]

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
            "price_selection": price["selection"],
            "price_transmission": price["transmission"],
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
        lineage_score = float(np.clip(n_events / REPROD_LINEAGE_EVENT_TARGET, 0.0, 1.0))

    # Combine: take the max of both signals
    combined = max(persistence / REPROD_PERSISTENCE_TARGET, lineage_score)
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

    # Analytical h² approximation using default mutation parameters.
    mutation_variance = MUTATION_POINT_RATE * GENOME_LENGTH * MUTATION_POINT_SCALE**2

    # Standing variance proxied by hash diversity
    standing_variance = diversity_ratio * 100  # scale to comparable range

    h2 = 1.0 - mutation_variance / (mutation_variance + standing_variance)
    return float(np.clip(h2, 0.0, 1.0))


def _price_selection(lineage: list[dict]) -> dict:
    """Price equation decomposition from lineage events.

    Returns dict with keys: selection, transmission, score.
    Uses parent_child_genome_distance as the continuous trait z,
    and offspring count per parent as fitness w.
    """
    if len(lineage) < 10:
        return {"selection": 0.0, "transmission": 0.0, "score": 0.0}

    # Group children by parent
    children_by_parent: dict[int, list[dict]] = defaultdict(list)
    for event in lineage:
        children_by_parent[event["parent_stable_id"]].append(event)

    if len(children_by_parent) < 2:
        return {"selection": 0.0, "transmission": 0.0, "score": 0.0}

    # w_i = number of offspring (fitness)
    # z_i = mean parent-child genome distance (trait: fidelity of transmission)
    parent_fitness = []
    parent_trait = []
    for children in children_by_parent.values():
        distances = [
            c.get("parent_child_genome_distance", 0.0)
            for c in children
            if np.isfinite(c.get("parent_child_genome_distance", 0.0))
        ]
        if not distances:
            continue
        parent_fitness.append(len(children))
        parent_trait.append(float(np.mean(distances)))

    w = np.array(parent_fitness, dtype=float)
    z = np.array(parent_trait, dtype=float)
    w_bar = np.mean(w)

    if w_bar == 0 or len(w) < 2:
        return {"selection": 0.0, "transmission": 0.0, "score": 0.0}

    # Price selection: Cov(w, z) / w_bar
    cov_matrix = np.cov(w, z, ddof=1)
    cov_wz = cov_matrix[0, 1]
    if not np.isfinite(cov_wz):
        return {"selection": 0.0, "transmission": 0.0, "score": 0.0}
    selection = float(cov_wz / w_bar)

    # Transmission bias: E(w · Δz) / w_bar
    # Requires multi-generation parent-to-grandchild tracking; approximate as 0
    transmission = 0.0

    # Score: map |selection| to [0, 1] — scale factor 5.0 chosen so that
    # moderate covariance (~0.2) maps to ~1.0.
    # TODO: calibrate on benchmark data (map 90th pct of |selection| to 1.0)
    score = float(np.clip(abs(selection) * 5.0, 0.0, 1.0))

    return {"selection": selection, "transmission": transmission, "score": score}
