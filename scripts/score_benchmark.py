"""Score benchmark runs with D1–D4 adapters and output a flat TSV score matrix.

Loads benchmark JSONs from experiments/benchmark/, runs all four definition
adapters on each (regime × seed × family), and outputs a TSV to stdout.

Usage:
    uv run python -m scripts.score_benchmark --data-dir experiments/benchmark \
        --seeds 0-4 --regimes E1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from adapters import score_all
from adapters.common import AdapterResult, discover_family_ids
from experiment_common import log, parse_regimes, parse_seed_range, safe_path

# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

# Ordered definition names
DEFINITIONS = ["D1", "D2", "D3", "D4"]


def build_score_row(
    run_summary: dict,
    *,
    regime: str,
    seed: int,
    family_id: int,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Score one family and return a flat dict row for the score matrix.

    Includes top-level score/pass columns and per-sub-criterion columns.
    """
    results = score_all(run_summary, family_id=family_id, thresholds=thresholds)

    row: dict = {
        "regime": regime,
        "seed": seed,
        "family_id": family_id,
    }

    for defn in DEFINITIONS:
        r: AdapterResult = results[defn]
        row[f"{defn}_score"] = r.score
        row[f"{defn}_pass"] = r.passes_threshold
        # Sub-criteria with definition prefix
        for crit_name, crit_val in r.criteria.items():
            row[f"{defn}_{crit_name}"] = crit_val

    return row


def score_run(
    run_summary: dict,
    *,
    regime: str,
    seed: int,
    thresholds: dict[str, float] | None = None,
) -> list[dict]:
    """Score all families in a single run. Returns a list of row dicts."""
    family_ids = discover_family_ids(run_summary)
    return [
        build_score_row(
            run_summary,
            regime=regime,
            seed=seed,
            family_id=fid,
            thresholds=thresholds,
        )
        for fid in family_ids
    ]


# ---------------------------------------------------------------------------
# TSV output
# ---------------------------------------------------------------------------


def _column_order(rows: list[dict]) -> list[str]:
    """Determine stable column order from row dicts."""
    if not rows:
        return []
    # Fixed prefix columns, then definition columns in order
    prefix = ["regime", "seed", "family_id"]
    seen: set[str] = set(prefix)
    rest: list[str] = []
    for defn in DEFINITIONS:
        for key in rows[0]:
            if key.startswith(f"{defn}_") and key not in seen:
                rest.append(key)
                seen.add(key)
    # Any remaining keys
    for key in rows[0]:
        if key not in seen:
            rest.append(key)
            seen.add(key)
    return prefix + rest


def format_tsv(rows: list[dict]) -> str:
    """Format score rows as a TSV string with header."""
    if not rows:
        return ""
    columns = _column_order(rows)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # Convert booleans to 0/1 for TSV
        out = {}
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, bool):
                out[col] = 1 if val else 0
            elif isinstance(val, float):
                out[col] = f"{val:.6f}"
            else:
                out[col] = val
        writer.writerow(out)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: score benchmark runs and output TSV to stdout."""
    parser = argparse.ArgumentParser(description="Score benchmark runs with D1–D4")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/benchmark"),
        help="Directory containing regime subdirectories with seed JSONs",
    )
    parser.add_argument("--seeds", default="0-99", help="Seed range (e.g. '0-4')")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5", help="Comma-separated regimes")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    seeds = parse_seed_range(args.seeds)
    regimes = parse_regimes(args.regimes)

    all_rows: list[dict] = []
    n_missing = 0

    for regime in regimes:
        regime_dir = safe_path(data_dir, regime)
        for seed in seeds:
            seed_file = regime_dir / f"seed_{seed:03d}.json"
            if not seed_file.exists():
                n_missing += 1
                log(f"  [skip] {regime}/seed_{seed:03d}.json not found")
                continue

            with open(seed_file) as f:
                run_summary = json.load(f)

            rows = score_run(run_summary, regime=regime, seed=seed)
            all_rows.extend(rows)
            log(f"  scored {regime}/seed_{seed:03d} ({len(rows)} families)")

    if n_missing:
        log(f"Warning: {n_missing} seed files not found")

    # TSV to stdout
    sys.stdout.write(format_tsv(all_rows))
    log(f"Score matrix: {len(all_rows)} rows ({len(all_rows) // 3} runs)")


if __name__ == "__main__":
    main()
