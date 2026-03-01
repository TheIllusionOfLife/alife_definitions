"""Benchmark harness for alife-defs definition comparison.

Generates the calibration dataset: 5 regimes × 3 families × seeds.
Mode B runs with F1 (full), F2 (Darwinian), F3 (autonomy) coexisting.

Output structure:
    experiments/benchmark/
    ├── E1/seed_000.json ... seed_099.json
    ├── E2/ E3/ E4/ E5/
    └── benchmark_manifest.json

Usage:
    uv run python -m scripts.experiment_benchmark --seeds 0-4 --regimes E1,E2 --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import alife_defs
from experiment_common import (
    FAMILY_PROFILES,
    experiment_output_dir,
    log,
    make_config_dict,
    safe_path,
)
from experiment_manifest import write_manifest

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

_REGIME_OVERRIDES: dict[str, dict] = {
    "E1": {},
    "E2": {"resource_regeneration_rate": 0.005, "world_size": 150.0},
    "E3": {"num_organisms": 80, "agents_per_organism": 30, "world_size": 80.0},
    "E4": {"sensing_noise_scale": 0.5},
    "E5": {"resource_patch_count": 4, "resource_patch_scale": 2.0},
}

ALL_REGIMES = list(_REGIME_OVERRIDES.keys())

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_regime_overrides(regime: str) -> dict:
    """Return the config overrides for a given regime label.

    Raises ValueError for unknown regimes.
    """
    if regime not in _REGIME_OVERRIDES:
        raise ValueError(f"Unknown regime '{regime}'. Expected one of: {ALL_REGIMES}")
    return dict(_REGIME_OVERRIDES[regime])


def run_benchmark(
    *,
    seeds: list[int],
    regimes: list[str],
    out_dir: Path | None = None,
    steps: int = 2000,
    sample_every: int = 50,
    resume: bool = False,
) -> dict[tuple[str, int], dict]:
    """Run the benchmark matrix and return results keyed by (regime, seed).

    Per-seed JSON files are written to ``out_dir/<regime>/seed_NNN.json``.
    With ``resume=True``, existing files are loaded instead of re-run.
    """
    if out_dir is None:
        out_dir = experiment_output_dir() / "benchmark"

    results: dict[tuple[str, int], dict] = {}

    for regime in regimes:
        overrides = get_regime_overrides(regime)
        regime_dir = safe_path(out_dir, regime)
        regime_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            seed_file = regime_dir / f"seed_{seed:03d}.json"
            key = (regime, seed)

            if resume and seed_file.exists():
                log(f"  [resume] {regime}/seed_{seed:03d} — loading existing")
                with open(seed_file) as f:
                    results[key] = json.load(f)
                continue

            t0 = time.perf_counter()
            config = _build_mode_b_config(seed, overrides)
            result_json = alife_defs.run_experiment_json(json.dumps(config), steps, sample_every)
            result = json.loads(result_json)
            result["regime_label"] = regime
            elapsed = time.perf_counter() - t0

            with open(seed_file, "w") as f:
                json.dump(result, f, indent=2)

            results[key] = result
            alive = result.get("final_alive_count", "?")
            log(f"  {regime}/seed_{seed:03d}  alive={alive}  {elapsed:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_mode_b_config(seed: int, regime_overrides: dict) -> dict:
    """Build a Mode B config with 3 family profiles and regime overrides."""
    config = make_config_dict(seed, regime_overrides)
    config["families"] = [dict(fp) for fp in FAMILY_PROFILES]
    # num_organisms must equal sum of family initial_counts
    config["num_organisms"] = sum(fp["initial_count"] for fp in FAMILY_PROFILES)
    return config


def _parse_seed_range(spec: str) -> list[int]:
    """Parse seed specification like '0-99' or '0,1,5-10'."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid seed specification: '{spec}'")
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"Invalid seed range '{part}': start must be <= end")
            seeds.extend(range(lo_i, hi_i + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"No seeds parsed from specification: '{spec}'")
    return sorted(set(seeds))


def _parse_regimes(spec: str) -> list[str]:
    """Parse regime specification like 'E1,E2,E3'."""
    regimes = [r.strip() for r in spec.split(",")]
    for r in regimes:
        if r not in _REGIME_OVERRIDES:
            raise ValueError(f"Unknown regime '{r}'. Expected one of: {ALL_REGIMES}")
    return regimes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark harness")
    parser.add_argument("--seeds", default="0-99", help="Seed range (e.g. '0-99', '0-4')")
    parser.add_argument("--regimes", default=",".join(ALL_REGIMES), help="Comma-separated regimes")
    parser.add_argument("--resume", action="store_true", help="Skip existing seed files")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per run")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample interval")
    args = parser.parse_args()

    seeds = _parse_seed_range(args.seeds)
    regimes = _parse_regimes(args.regimes)
    out_dir = experiment_output_dir() / "benchmark"

    n_runs = len(regimes) * len(seeds)
    log(f"Benchmark: {len(regimes)} regimes x {len(seeds)} seeds = {n_runs} runs")
    log(f"Output: {out_dir}")

    total_start = time.perf_counter()
    results = run_benchmark(
        seeds=seeds,
        regimes=regimes,
        out_dir=out_dir,
        steps=args.steps,
        sample_every=args.sample_every,
        resume=args.resume,
    )

    # Write manifest — use _build_mode_b_config so families are included
    # in the digest, matching the actual executed config.
    base_config = _build_mode_b_config(0, {})
    write_manifest(
        out_dir / "benchmark_manifest.json",
        experiment_name="benchmark",
        steps=args.steps,
        sample_every=args.sample_every,
        seeds=seeds,
        base_config=base_config,
        condition_overrides={r: get_regime_overrides(r) for r in regimes},
    )

    elapsed = time.perf_counter() - total_start
    log(f"Benchmark complete: {len(results)} runs in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
