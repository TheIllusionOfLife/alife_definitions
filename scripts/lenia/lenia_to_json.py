"""Map Lenia simulation traces to the alife_defs RunSummary JSON schema.

Mapping (specified a priori, invariant to scale/discretization):
  - mass (sum of cell values) → energy_mean
  - spatial compactness (inverse bounding radius, normalized) → boundary_mean
  - pattern entropy (Shannon entropy of grid) → waste_mean
  - connected components above mass threshold → alive_count
  - 0 (no explicit reproduction) → birth_count
  - morphology parameter variation → genome_diversity
  - rolling autocorrelation of mass → maturity_mean

This module provides the mapping functions and schema conversion.
The actual Lenia simulation is handled by lenia_runner.py.
"""

from __future__ import annotations

import numpy as np


def _spatial_compactness(grid: np.ndarray) -> float:
    """Compute spatial compactness as inverse of bounding radius, normalized to [0,1]."""
    if grid.sum() < 1e-10:
        return 0.0
    ys, xs = np.where(grid > 0.01)
    if len(xs) == 0:
        return 0.0
    cy, cx = np.mean(ys), np.mean(xs)
    max_dist = np.max(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2))
    max_possible = np.sqrt(grid.shape[0] ** 2 + grid.shape[1] ** 2) / 2
    if max_possible == 0:
        return 0.0
    return float(np.clip(1.0 - max_dist / max_possible, 0.0, 1.0))


def _pattern_entropy(grid: np.ndarray, n_bins: int = 20) -> float:
    """Compute Shannon entropy of the grid value distribution, normalized."""
    flat = grid.flatten()
    if flat.sum() < 1e-10:
        return 0.0
    hist, _ = np.histogram(flat, bins=n_bins, range=(0, 1), density=True)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    nonzero = hist[hist > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    max_entropy = np.log2(n_bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0


def _connected_components(grid: np.ndarray, threshold: float = 0.1) -> int:
    """Count connected components above mass threshold using simple BFS."""
    binary = grid > threshold
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    rows, cols = binary.shape

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] and not visited[r, c]:
                count += 1
                # BFS flood fill
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if binary[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
    return count


def _rolling_autocorrelation(values: list[float], window: int = 10) -> float:
    """Compute rolling autocorrelation (lag-1) of a time series as maturity proxy."""
    if len(values) < window + 1:
        return 0.0
    arr = np.array(values)
    autocorrs = []
    for i in range(len(arr) - window):
        seg = arr[i : i + window]
        if np.std(seg) < 1e-10:
            autocorrs.append(1.0)
            continue
        r = np.corrcoef(seg[:-1], seg[1:])[0, 1]
        autocorrs.append(float(r) if not np.isnan(r) else 0.0)
    return float(np.clip(np.mean(autocorrs), 0.0, 1.0))


def grid_sequence_to_run_summary(
    grids: list[np.ndarray],
    *,
    creature_name: str = "unknown",
    seed: int = 0,
    sample_every: int = 1,
) -> dict:
    """Convert a sequence of Lenia grid snapshots to RunSummary JSON format.

    Args:
        grids: List of 2D numpy arrays representing Lenia grid states.
        creature_name: Name of the Lenia creature (for metadata).
        seed: Random seed used for the simulation.
        sample_every: Step interval between consecutive grids.

    Returns:
        Dict conforming to the RunSummary schema used by adapters.score_all().
    """
    samples: list[dict] = []
    mass_history: list[float] = []

    for i, grid in enumerate(grids):
        mass = float(grid.sum())
        mass_history.append(mass)

        step = i * sample_every
        n_alive = _connected_components(grid)
        compactness = _spatial_compactness(grid)
        entropy = _pattern_entropy(grid)

        sample = {
            "step": step,
            "alive_count": n_alive,
            "energy_mean": mass / max(grid.size, 1),
            "waste_mean": entropy,
            "boundary_mean": compactness,
            "birth_count": 0,  # Lenia has no explicit reproduction
            "death_count": 0,
            "population_size": n_alive,
            "mean_generation": 0.0,
            "mean_genome_drift": 0.0,
            "genome_diversity": 0.0,  # Will be set below per-creature
            "maturity_mean": _rolling_autocorrelation(mass_history),
            "energy_std": float(np.std(grid)),
            "waste_std": 0.0,
            "boundary_std": 0.0,
            "mean_age": float(i * sample_every),
            "max_generation": 0,
            "family_breakdown": [
                {
                    "family_id": 0,
                    "alive_count": n_alive,
                    "population_size": n_alive,
                    "energy_mean": mass / max(grid.size, 1),
                    "waste_mean": entropy,
                    "boundary_mean": compactness,
                    "birth_count": 0,
                    "death_count": 0,
                    "mean_generation": 0.0,
                    "mean_genome_drift": 0.0,
                    "genome_diversity": 0.0,
                    "maturity_mean": _rolling_autocorrelation(mass_history),
                }
            ],
        }
        samples.append(sample)

    return {
        "samples": samples,
        "lineage_events": [],  # No reproduction in Lenia
        "final_alive_count": samples[-1]["alive_count"] if samples else 0,
        "regime_label": "lenia",
        "metadata": {
            "creature_name": creature_name,
            "seed": seed,
            "substrate": "lenia",
        },
    }
