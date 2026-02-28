"""D4 — Information Maintenance adapter.

Life as a system where heritable information is not just present but
*causally maintains itself*: the genome must predict fitness, and that
prediction must persist across generations.

Sub-scores:
  S_info_present — Non-trivial heritable information exists
  S_info_causal — Genome information predicts fitness (TE + Spearman)
  S_info_preserved — Information persists across generations (hash stability)

Aggregate: weighted average (2× weight on causal component).
"""

from __future__ import annotations

import numpy as np
from analyses.coupling.transfer_entropy import transfer_entropy_lag1
from scipy import stats

from .common import (
    AdapterResult,
    discover_family_ids,
    extract_family_lineage,
    extract_family_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.3
TE_BINS = 5  # 5 bins viable for n≈200 with xcorr complement
TE_PERMS = 400
TE_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_d4(
    run_summary: dict,
    *,
    family_id: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> AdapterResult:
    """Score a family against D4 (information maintenance)."""
    rng = np.random.default_rng(4026 + family_id)
    series = extract_family_series(run_summary, family_id)
    lineage = extract_family_lineage(run_summary, family_id)

    # Extract all families for cross-family baseline
    family_ids = discover_family_ids(run_summary)
    all_families = {fid: extract_family_series(run_summary, fid) for fid in family_ids}

    s_present = _score_info_present(series, all_families)
    s_causal = _score_info_causal(series, rng)
    s_preserved = _score_info_preserved(lineage)

    criteria = {
        "S_info_present": s_present,
        "S_info_causal": s_causal,
        "S_info_preserved": s_preserved,
    }

    # Weighted average: 2× weight on causal
    aggregate = (s_present + 2.0 * s_causal + s_preserved) / 4.0

    return AdapterResult(
        definition="D4",
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


def _score_info_present(
    series: dict[str, np.ndarray],
    all_families: dict[int, dict[str, np.ndarray]],
) -> float:
    """S_info_present — Non-trivial heritable information exists."""
    diversity = series["genome_diversity"]
    drift = series["mean_genome_drift"]

    if len(diversity) == 0:
        return 0.0

    # Must have non-zero drift (genome is not frozen)
    if float(np.mean(drift)) <= 0:
        return 0.0

    # Compare final diversity to median across all families
    final_diversity = diversity[-1]
    all_final = [
        fs["genome_diversity"][-1]
        for fs in all_families.values()
        if len(fs["genome_diversity"]) > 0
    ]
    baseline = float(np.median(all_final)) if all_final else 1.0

    if baseline <= 0:
        return 1.0 if final_diversity > 0 else 0.0

    return float(np.clip(final_diversity / baseline, 0.0, 1.0))


def _score_info_causal(
    series: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> float:
    """S_info_causal — Genome information predicts fitness."""
    genome_div = series["genome_diversity"]
    alive = series["alive_count"]

    if len(genome_div) < 4 or np.std(genome_div) == 0 or np.std(alive) == 0:
        return 0.0

    # Spearman correlation
    rho, _ = stats.spearmanr(genome_div, alive)
    if np.isnan(rho):
        rho = 0.0

    # Transfer entropy: genome_diversity → alive_count
    te_result = transfer_entropy_lag1(
        genome_div, alive, bins=TE_BINS, permutations=TE_PERMS, rng=rng
    )
    te_bonus = 0.3 if (te_result and te_result["p_value"] < TE_ALPHA) else 0.0

    return float(np.clip(abs(rho) + te_bonus, 0.0, 1.0))


def _score_info_preserved(lineage: list[dict]) -> float:
    """S_info_preserved — Information persists across generations.

    Measures intergenerational genome hash stability:
    mean similarity of genome_hash between consecutive generations.
    """
    if not lineage:
        return 0.0

    # Group hashes by generation
    gen_hashes: dict[int, list[int]] = {}
    for event in lineage:
        gen = event.get("generation", 0)
        genome_hash = event.get("genome_hash", 0)
        gen_hashes.setdefault(gen, []).append(genome_hash)

    # Sort generations
    sorted_gens = sorted(gen_hashes.keys())
    if len(sorted_gens) < 2:
        # Single generation — can't measure cross-generation preservation
        return 0.3  # minimal evidence of preservation

    # Compute similarity between consecutive generations
    similarities: list[float] = []
    for g1, g2 in zip(sorted_gens[:-1], sorted_gens[1:], strict=True):
        hashes1 = set(gen_hashes[g1])
        hashes2 = set(gen_hashes[g2])
        if not hashes1 or not hashes2:
            continue
        # Jaccard-like similarity: how many hashes are shared
        intersection = len(hashes1 & hashes2)
        union = len(hashes1 | hashes2)
        if union > 0:
            similarities.append(intersection / union)

    if not similarities:
        return 0.1  # lineage exists but no hash overlap measurable

    mean_sim = float(np.mean(similarities))
    # Map from [0, 1] where 0.5 would be random → clip(2*(sim - 0.5), 0, 1)
    # But hash similarity of 0 is expected for mutated genomes, so use raw similarity
    return float(np.clip(mean_sim, 0.0, 1.0))
