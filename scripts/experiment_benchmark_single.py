"""Single-family control harness for competition confound analysis.

Runs each family profile in isolation (Mode B with 1 family) to determine
whether family ordering is due to capability deficit or competition loss.

Output structure:
    experiments/benchmark_single/
    ├── F1/E1/seed_000.json ... seed_199.json
    ├── F1/E2/ ...
    ├── F2/E1/ ...
    └── benchmark_single_manifest.json

Usage:
    uv run python -m scripts.experiment_benchmark_single \
        --seeds 0-4 --regimes E1,E2 --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import alife_defs
from experiment_benchmark import ALL_REGIMES, get_regime_overrides
from experiment_common import (
    FAMILY_PROFILES,
    experiment_output_dir,
    log,
    make_config_dict,
    safe_path,
)
from experiment_manifest import write_manifest

# Family labels for directory naming
FAMILY_LABELS = ["F1", "F2", "F3"]


def _build_single_family_config(
    seed: int,
    family_profile: dict,
    regime_overrides: dict,
) -> dict:
    """Single-family Mode B: one-entry families list preserves family_breakdown output."""
    config = make_config_dict(seed, regime_overrides)
    config["families"] = [dict(family_profile)]
    config["num_organisms"] = family_profile["initial_count"]
    return config


def run_single_family_benchmark(
    *,
    seeds: list[int],
    regimes: list[str],
    out_dir: Path | None = None,
    steps: int = 2000,
    sample_every: int = 10,
    resume: bool = False,
) -> dict[tuple[str, str, int], dict]:
    """Run single-family isolation experiments.

    Returns results keyed by (family_label, regime, seed).
    """
    if out_dir is None:
        out_dir = experiment_output_dir() / "benchmark_single"

    results: dict[tuple[str, str, int], dict] = {}

    for fam_idx, (fam_label, fam_profile) in enumerate(
        zip(FAMILY_LABELS, FAMILY_PROFILES, strict=True)
    ):
        for regime in regimes:
            overrides = get_regime_overrides(regime)
            regime_dir = safe_path(out_dir, fam_label, regime)
            regime_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                seed_file = regime_dir / f"seed_{seed:03d}.json"
                key = (fam_label, regime, seed)

                if resume and seed_file.exists():
                    log(f"  [resume] {fam_label}/{regime}/seed_{seed:03d}")
                    with open(seed_file) as f:
                        results[key] = json.load(f)
                    continue

                t0 = time.perf_counter()
                config = _build_single_family_config(seed, fam_profile, overrides)
                result_json = alife_defs.run_experiment_json(
                    json.dumps(config), steps, sample_every
                )
                result = json.loads(result_json)
                result["regime_label"] = regime
                result["family_label"] = fam_label
                result["family_index"] = fam_idx
                elapsed = time.perf_counter() - t0

                with open(seed_file, "w") as f:
                    json.dump(result, f, indent=2)

                results[key] = result
                alive = result.get("final_alive_count", "?")
                log(f"  {fam_label}/{regime}/seed_{seed:03d}  alive={alive}  {elapsed:.2f}s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-family control experiments")
    parser.add_argument("--seeds", default="0-199", help="Seed range")
    parser.add_argument("--regimes", default=",".join(ALL_REGIMES), help="Comma-separated regimes")
    parser.add_argument("--resume", action="store_true", help="Skip existing seed files")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per run")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample interval")
    args = parser.parse_args()

    from experiment_common import parse_seed_range

    seeds = parse_seed_range(args.seeds)
    regimes = [r.strip() for r in args.regimes.split(",")]
    out_dir = experiment_output_dir() / "benchmark_single"

    n_runs = len(FAMILY_LABELS) * len(regimes) * len(seeds)
    log(
        f"Single-family: {len(FAMILY_LABELS)} families × "
        f"{len(regimes)} regimes × {len(seeds)} seeds = {n_runs} runs"
    )
    log(f"Output: {out_dir}")

    total_start = time.perf_counter()
    results = run_single_family_benchmark(
        seeds=seeds,
        regimes=regimes,
        out_dir=out_dir,
        steps=args.steps,
        sample_every=args.sample_every,
        resume=args.resume,
    )

    # Write manifest
    base_config = _build_single_family_config(0, dict(FAMILY_PROFILES[0]), {})
    write_manifest(
        out_dir / "benchmark_single_manifest.json",
        experiment_name="benchmark_single",
        steps=args.steps,
        sample_every=args.sample_every,
        seeds=seeds,
        base_config=base_config,
        condition_overrides={r: get_regime_overrides(r) for r in regimes},
    )

    elapsed = time.perf_counter() - total_start
    log(f"Single-family complete: {len(results)} runs in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
