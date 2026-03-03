"""Lenia simulation runner for cross-substrate validation.

Implements a minimal Lenia simulation (Smooth Life variant) for testing
the D2-D3 reversal hypothesis. Uses the official Lenia update rule:
  A(t+1) = clip(A(t) + dt * G(K*A(t)), 0, 1)
where K is a ring-shaped kernel and G is a Gaussian growth function.

Pre-registered hypothesis H_Lenia:
  "In a substrate without explicit reproduction (Lenia), D3 (organizational
  closure) will score higher than D2 (Darwinian evolution)."

Usage:
    uv run python -m scripts.lenia.lenia_runner --creature orbium --seeds 0-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .lenia_to_json import grid_sequence_to_run_summary

# ---------------------------------------------------------------------------
# Lenia kernel and growth functions
# ---------------------------------------------------------------------------

# Known stable creatures: (mu_k, sigma_k, mu_g, sigma_g, R)
# Parameters from Bert Chan's original Lenia paper (2019)
CREATURES = {
    "orbium": {"mu_k": 0.5, "sigma_k": 0.15, "mu_g": 0.15, "sigma_g": 0.015, "R": 13},
    "geminium": {"mu_k": 0.5, "sigma_k": 0.15, "mu_g": 0.14, "sigma_g": 0.014, "R": 10},
    "scutium": {"mu_k": 0.5, "sigma_k": 0.15, "mu_g": 0.16, "sigma_g": 0.016, "R": 15},
}


def _ring_kernel(R: int, mu: float, sigma: float, grid_size: int) -> np.ndarray:
    """Create a ring-shaped Lenia kernel in frequency domain."""
    mid = grid_size // 2
    y, x = np.ogrid[-mid : grid_size - mid, -mid : grid_size - mid]
    r = np.sqrt(x * x + y * y) / R
    kernel = np.exp(-((r - mu) ** 2) / (2 * sigma * sigma))
    kernel[r > 1] = 0
    # Normalize
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def _growth(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian growth function G(u) = 2 * exp(-((u-mu)/sigma)^2 / 2) - 1."""
    return 2.0 * np.exp(-((u - mu) ** 2) / (2 * sigma * sigma)) - 1.0


def _init_orbium(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize a Lenia grid with a centered blob (Orbium-like initial condition)."""
    grid = np.zeros((grid_size, grid_size), dtype=float)
    mid = grid_size // 2
    r = grid_size // 8
    y, x = np.ogrid[-mid : grid_size - mid, -mid : grid_size - mid]
    dist = np.sqrt(x * x + y * y)
    grid[dist < r] = 0.8 + 0.2 * rng.random(grid[dist < r].shape)
    return grid


def run_lenia(
    creature: str = "orbium",
    *,
    grid_size: int = 128,
    steps: int = 500,
    dt: float = 0.1,
    sample_every: int = 5,
    seed: int = 0,
) -> dict:
    """Run a Lenia simulation and return a RunSummary-compatible dict.

    Args:
        creature: Name from CREATURES dict.
        grid_size: Size of the square grid.
        steps: Number of simulation steps.
        dt: Time step size.
        sample_every: Record grid every N steps.
        seed: Random seed.

    Returns:
        RunSummary dict compatible with adapters.score_all().
    """
    params = CREATURES[creature]
    rng = np.random.default_rng(seed)

    # Initialize grid
    grid = _init_orbium(grid_size, rng)

    # Pre-compute kernel FFT
    kernel = _ring_kernel(params["R"], params["mu_k"], params["sigma_k"], grid_size)
    kernel_fft = np.fft.fft2(np.fft.fftshift(kernel))

    grids: list[np.ndarray] = []
    if sample_every > 0:
        grids.append(grid.copy())

    for step in range(1, steps + 1):
        # Lenia update: convolution in frequency domain
        grid_fft = np.fft.fft2(grid)
        u = np.real(np.fft.ifft2(grid_fft * kernel_fft))
        growth = _growth(u, params["mu_g"], params["sigma_g"])
        grid = np.clip(grid + dt * growth, 0.0, 1.0)

        if sample_every > 0 and step % sample_every == 0:
            grids.append(grid.copy())

    return grid_sequence_to_run_summary(
        grids,
        creature_name=creature,
        seed=seed,
        sample_every=sample_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lenia simulation")
    parser.add_argument("--creature", default="orbium", choices=list(CREATURES.keys()))
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/lenia"))
    args = parser.parse_args()

    from experiment_common import log, parse_seed_range

    seeds = parse_seed_range(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Lenia: creature={args.creature}, seeds={args.seeds}")

    for seed in seeds:
        result = run_lenia(
            args.creature,
            grid_size=args.grid_size,
            steps=args.steps,
            sample_every=args.sample_every,
            seed=seed,
        )
        out_path = args.output_dir / f"{args.creature}_seed_{seed:03d}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"  {out_path}: alive={result['final_alive_count']}")

    # Score with D1-D4 adapters
    log("Scoring with D1-D4 adapters...")
    from adapters import score_all
    from adapters.common import discover_family_ids

    d2_scores: list[float] = []
    d3_scores: list[float] = []

    for seed in seeds:
        path = args.output_dir / f"{args.creature}_seed_{seed:03d}.json"
        with open(path) as f:
            run = json.load(f)
        for fid in discover_family_ids(run):
            result = score_all(run, family_id=fid)
            d2_scores.append(result["D2"].score)
            d3_scores.append(result["D3"].score)
            log(
                f"  seed={seed} fid={fid}: "
                f"D1={result['D1'].score:.3f} D2={result['D2'].score:.3f} "
                f"D3={result['D3'].score:.3f} D4={result['D4'].score:.3f}"
            )

    # H_Lenia test: D3 > D2?
    if d2_scores and d3_scores:
        d2_arr = np.array(d2_scores)
        d3_arr = np.array(d3_scores)
        mean_d2 = float(np.mean(d2_arr))
        mean_d3 = float(np.mean(d3_arr))
        # Cohen's d for paired comparison
        diff = d3_arr - d2_arr
        d_cohen = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
        log(f"\nH_Lenia: D3 mean={mean_d3:.3f}, D2 mean={mean_d2:.3f}, Cohen's d={d_cohen:.3f}")
        log(f"  D3 > D2: {mean_d3 > mean_d2}")


if __name__ == "__main__":
    main()
