"""Bootstrap confidence intervals for agreement metrics and predictive validity.

Block-bootstraps over seeds to compute 95% CIs for pairwise Cohen's kappa,
Spearman rho, Fleiss' kappa, and ROC-AUC values.

Usage:
    uv run python scripts/analyze_bootstrap_ci.py \
        experiments/score_matrix_cal.tsv \
        experiments/benchmark/ \
        -o experiments/bootstrap_ci.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

DEFINITIONS = ["D1", "D2", "D3", "D4"]
N_BOOT = 2000
ALPHA = 0.05
RNG_SEED = 2026


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Compute Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return 0.0
    tp = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    tn = sum(1 for x, y in zip(a, b, strict=True) if not x and not y)
    fp = sum(1 for x, y in zip(a, b, strict=True) if not x and y)
    fn = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe == 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(pass_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa. pass_matrix: (n_subjects, n_raters) bool."""
    n, k = pass_matrix.shape
    if n == 0 or k < 2:
        return 0.0
    # Count of "pass" per rater category (pass/fail = 2 categories)
    counts = np.zeros((n, 2), dtype=float)
    counts[:, 1] = pass_matrix.sum(axis=1)  # pass count
    counts[:, 0] = k - counts[:, 1]  # fail count

    p_j = counts.sum(axis=0) / (n * k)
    pe = float(np.sum(p_j**2))

    p_i = np.sum(counts**2, axis=1) - k
    p_i = p_i / (k * (k - 1))
    po = float(np.mean(p_i))

    if pe == 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def load_score_matrix(path: Path) -> list[dict]:
    """Load TSV score matrix into list of row dicts."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def bootstrap_agreement(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    """Block-bootstrap over seeds for κ, ρ, and Fleiss' κ CIs."""
    rng = np.random.default_rng(RNG_SEED)

    # Group rows by seed
    seed_groups: dict[int, list[dict]] = {}
    for row in rows:
        seed = int(row["seed"])
        seed_groups.setdefault(seed, []).append(row)
    seeds = sorted(seed_groups.keys())
    n_seeds = len(seeds)
    if n_seeds == 0:
        raise ValueError(
            "No seeds found in score matrix for agreement bootstrap. "
            "Check that the input TSV contains rows with a 'seed' column."
        )

    pairs = list(combinations(DEFINITIONS, 2))

    # Storage for bootstrap samples
    boot_kappa = {f"{a}_{b}": np.empty(n_boot) for a, b in pairs}
    boot_rho = {f"{a}_{b}": np.empty(n_boot) for a, b in pairs}
    boot_fleiss = np.empty(n_boot)

    for b in range(n_boot):
        # Resample seeds with replacement
        sampled_seeds = rng.choice(seeds, size=n_seeds, replace=True)
        boot_rows = []
        for s in sampled_seeds:
            boot_rows.extend(seed_groups[s])

        # Extract scores and passes
        scores = {d: [float(r[f"{d}_score"]) for r in boot_rows] for d in DEFINITIONS}
        passes = {d: [r[f"{d}_pass"] == "1" for r in boot_rows] for d in DEFINITIONS}

        # Pairwise metrics
        for di, dj in pairs:
            key = f"{di}_{dj}"
            boot_kappa[key][b] = cohens_kappa(passes[di], passes[dj])
            if np.std(scores[di]) > 0 and np.std(scores[dj]) > 0:
                rho_val, _ = stats.spearmanr(scores[di], scores[dj])
                boot_rho[key][b] = float(rho_val) if not np.isnan(rho_val) else 0.0
            else:
                boot_rho[key][b] = 0.0

        # Fleiss' kappa
        pass_mat = np.array(
            [[passes[d][i] for d in DEFINITIONS] for i in range(len(boot_rows))],
            dtype=float,
        )
        boot_fleiss[b] = fleiss_kappa(pass_mat)

    # Compute CIs
    lo_pct = 100 * ALPHA / 2
    hi_pct = 100 * (1 - ALPHA / 2)

    result: dict = {"pairwise": {}, "fleiss_kappa": {}}
    for di, dj in pairs:
        key = f"{di}_{dj}"
        result["pairwise"][key] = {
            "kappa_ci": [
                round(float(np.percentile(boot_kappa[key], lo_pct)), 2),
                round(float(np.percentile(boot_kappa[key], hi_pct)), 2),
            ],
            "rho_ci": [
                round(float(np.percentile(boot_rho[key], lo_pct)), 2),
                round(float(np.percentile(boot_rho[key], hi_pct)), 2),
            ],
        }

    result["fleiss_kappa"] = {
        "ci": [
            round(float(np.percentile(boot_fleiss, lo_pct)), 2),
            round(float(np.percentile(boot_fleiss, hi_pct)), 2),
        ],
    }

    return result


def bootstrap_roc_auc(
    data_dir: Path,
    test_seeds: list[int],
    regimes: list[str],
    n_boot: int = N_BOOT,
) -> dict:
    """Block-bootstrap over test seeds for ROC-AUC CIs.

    Pre-computes all scores once, then resamples seed indices for each
    bootstrap iteration (avoids re-scoring on every iteration).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from adapters import score_all
    from adapters.common import discover_family_ids, extract_family_series
    from analyze_predictive import _make_labels, roc_auc_score

    rng = np.random.default_rng(RNG_SEED)

    # np.trapezoid was added in NumPy 2.0; np.trapz was removed in NumPy 2.0.
    try:
        _trapezoid = np.trapezoid  # type: ignore[attr-defined]
    except AttributeError:
        _trapezoid = np.trapz  # type: ignore[attr-defined]

    # Pre-compute all test scores and targets, grouped by seed
    print("Loading and scoring test runs...", file=sys.stderr)
    seed_data: dict[int, dict] = {}  # seed -> {scores: {D: [float]}, targets: [float]}

    for regime in regimes:
        regime_dir = data_dir / regime
        for seed in test_seeds:
            path = regime_dir / f"seed_{seed:03d}.json"
            if not path.exists():
                continue
            with open(path) as f:
                run = json.load(f)

            family_ids = discover_family_ids(run)
            for fid in family_ids:
                result = score_all(run, family_id=fid)

                if seed not in seed_data:
                    seed_data[seed] = {d: [] for d in DEFINITIONS}
                    seed_data[seed]["_targets"] = []

                for d in DEFINITIONS:
                    seed_data[seed][d].append(result[d].score)

                # alive_auc target
                series = extract_family_series(run, fid)
                alive = series["alive_count"]
                if len(alive) >= 2:
                    n = len(alive)
                    start = max(0, n - int(n * 0.3))
                    tail = alive[start:]
                    auc = float(_trapezoid(tail, np.arange(len(tail)))) if len(tail) >= 2 else 0.0
                else:
                    auc = 0.0
                seed_data[seed]["_targets"].append(auc)

        print(f"  {regime} done", file=sys.stderr)

    available_seeds = sorted(seed_data.keys())
    n_avail = len(available_seeds)
    if n_avail == 0:
        raise ValueError(
            f"No test seed data found in {data_dir} for regimes {regimes}. "
            "Check that the data directory and seed range are correct."
        )
    print(f"Scored {n_avail} seeds, bootstrapping {n_boot} iterations...", file=sys.stderr)

    # Compute labels ONCE from full dataset so the estimand is fixed
    # (matches the point-estimate procedure in analyze_predictive.py).
    full_targets: list[float] = []
    for s in available_seeds:
        full_targets.extend(seed_data[s]["_targets"])
    full_labels, _ = _make_labels(full_targets)

    # Build per-seed label slices for fast reassembly during resampling
    seed_labels: dict[int, list[bool]] = {}
    offset = 0
    for s in available_seeds:
        n_items = len(seed_data[s]["_targets"])
        seed_labels[s] = full_labels[offset : offset + n_items]
        offset += n_items

    boot_auc = {d: np.empty(n_boot) for d in DEFINITIONS}

    b = 0
    max_attempts = n_boot * 10  # safety cap to avoid infinite loop
    attempts = 0
    while b < n_boot and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(available_seeds, size=n_avail, replace=True)

        # Gather scores and FIXED labels from sampled seeds
        all_scores = {d: [] for d in DEFINITIONS}
        all_labels: list[bool] = []
        for s in sampled:
            sd = seed_data[s]
            for d in DEFINITIONS:
                all_scores[d].extend(sd[d])
            all_labels.extend(seed_labels[s])

        # Reject degenerate draws (single-class labels) to avoid
        # injecting chance-level AUC = 0.5 into the distribution.
        if len(set(all_labels)) < 2:
            continue

        for d in DEFINITIONS:
            boot_auc[d][b] = roc_auc_score(all_labels, all_scores[d])

        b += 1
        if b % 500 == 0:
            print(f"  {b}/{n_boot}", file=sys.stderr)

    if b < n_boot:
        print(
            f"WARNING: only {b}/{n_boot} valid bootstrap iterations "
            f"(too many single-class resamples)",
            file=sys.stderr,
        )

    lo_pct = 100 * ALPHA / 2
    hi_pct = 100 * (1 - ALPHA / 2)

    result = {}
    for d in DEFINITIONS:
        filled = boot_auc[d][:b]
        result[d] = {
            "auc_ci": [
                round(float(np.percentile(filled, lo_pct)), 2),
                round(float(np.percentile(filled, hi_pct)), 2),
            ]
            if len(filled) > 0
            else [float("nan"), float("nan")],
        }
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Bootstrap CIs for paper metrics")
    parser.add_argument("score_matrix", type=Path, help="Score matrix TSV")
    parser.add_argument(
        "data_dir", type=Path, nargs="?", help="Benchmark data dir (for ROC-AUC CIs)"
    )
    parser.add_argument("--test-seeds", default="100-199", help="Test seed range")
    parser.add_argument("--regimes", default="E1,E2,E3,E4,E5")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--agreement-only", action="store_true", help="Skip ROC-AUC bootstrap")
    args = parser.parse_args()

    print("=== Agreement bootstrap ===", file=sys.stderr)
    rows = load_score_matrix(args.score_matrix)
    agreement = bootstrap_agreement(rows)

    result = {"agreement": agreement}

    if args.data_dir and not args.agreement_only:
        print("\n=== ROC-AUC bootstrap ===", file=sys.stderr)

        def parse_range(s: str) -> list[int]:
            lo, hi = s.split("-")
            return list(range(int(lo), int(hi) + 1))

        test = parse_range(args.test_seeds)
        regimes = args.regimes.split(",")
        roc = bootstrap_roc_auc(args.data_dir.resolve(), test, regimes)
        result["roc_auc"] = roc

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
