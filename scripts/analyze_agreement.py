"""Pairwise agreement analysis among D1–D4 definition scores.

Input: score matrix rows (list of dicts from score_benchmark.score_run).
Output: JSON with pairwise κ, ρ, percent agreement, disagreement characterization,
        and aggregate Fleiss' κ.

Usage:
    uv run python scripts/analyze_agreement.py experiments/benchmark/score_matrix.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

DEFINITIONS = ["D1", "D2", "D3", "D4"]


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Compute Cohen's kappa for two binary raters.

    Returns 0.0 when expected agreement is 1.0 (degenerate case:
    both raters assign the same label to every item).
    """
    n = len(a)
    if n == 0:
        return 0.0

    # Build confusion matrix
    tp = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    tn = sum(1 for x, y in zip(a, b, strict=True) if not x and not y)
    fp = sum(1 for x, y in zip(a, b, strict=True) if not x and y)
    fn = sum(1 for x, y in zip(a, b, strict=True) if x and not y)

    po = (tp + tn) / n  # observed agreement
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)  # expected

    if pe == 1.0:
        return 0.0  # degenerate: all same label
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------


def compute_pairwise(
    scores_i: list[float],
    scores_j: list[float],
    passes_i: list[bool],
    passes_j: list[bool],
) -> dict:
    """Compute pairwise agreement metrics between two definitions."""
    kappa = cohens_kappa(passes_i, passes_j)

    # Spearman on graded scores
    if len(scores_i) >= 3 and np.std(scores_i) > 0 and np.std(scores_j) > 0:
        rho, p_rho = stats.spearmanr(scores_i, scores_j)
        if np.isnan(rho):
            rho = 0.0
            p_rho = 1.0
    else:
        rho = 0.0
        p_rho = 1.0

    # Percent agreement (binary)
    n = len(passes_i)
    agree = sum(1 for a, b in zip(passes_i, passes_j, strict=True) if a == b)
    pct = agree / n if n > 0 else 0.0

    return {
        "cohens_kappa": float(kappa),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "percent_agreement": float(pct),
    }


# ---------------------------------------------------------------------------
# Disagreement characterization
# ---------------------------------------------------------------------------


def characterize_disagreements(rows: list[dict]) -> dict:
    """Characterize disagreements between two definitions.

    Each row must have keys: regime, family_id, Di_pass, Dj_pass.
    """
    i_acc_j_rej = 0
    j_acc_i_rej = 0
    total = 0

    by_regime: dict[str, dict] = defaultdict(lambda: {"i_accepts": 0, "j_accepts": 0, "total": 0})
    by_family: dict[int, dict] = defaultdict(lambda: {"i_accepts": 0, "j_accepts": 0, "total": 0})

    for row in rows:
        di = row["Di_pass"]
        dj = row["Dj_pass"]
        if di == dj:
            continue
        total += 1
        regime = row["regime"]
        fid = row["family_id"]

        if di and not dj:
            i_acc_j_rej += 1
            by_regime[regime]["i_accepts"] += 1
            by_family[fid]["i_accepts"] += 1
        else:
            j_acc_i_rej += 1
            by_regime[regime]["j_accepts"] += 1
            by_family[fid]["j_accepts"] += 1

        by_regime[regime]["total"] += 1
        by_family[fid]["total"] += 1

    return {
        "i_accepts_j_rejects": i_acc_j_rej,
        "j_accepts_i_rejects": j_acc_i_rej,
        "total_disagreements": total,
        "by_regime": dict(by_regime),
        "by_family": {k: dict(v) for k, v in by_family.items()},
    }


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Compute Fleiss' kappa for multiple raters.

    Args:
        ratings: (n_items, n_categories) matrix where ratings[i][j] = number
                 of raters who assigned item i to category j.

    Returns:
        Fleiss' kappa statistic.
    """
    n_items, n_cats = ratings.shape
    n_raters = int(ratings[0].sum())

    if n_items == 0 or n_raters <= 1:
        return 0.0

    # Proportion of assignments to each category
    p_j = ratings.sum(axis=0) / (n_items * n_raters)

    # Per-item agreement
    p_i = (np.sum(ratings**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)

    # Expected agreement
    pe = np.sum(p_j**2)

    if pe == 1.0:
        return 0.0
    return float((p_bar - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze_agreement(score_rows: list[dict]) -> dict:
    """Run full agreement analysis on score matrix rows.

    Args:
        score_rows: List of row dicts with D1_score, D1_pass, etc.

    Returns:
        Dict with pairwise and aggregate agreement metrics.
    """
    result: dict = {"pairwise": {}, "aggregate": {}}

    # Extract scores and passes per definition
    def_scores: dict[str, list[float]] = {d: [] for d in DEFINITIONS}
    def_passes: dict[str, list[bool]] = {d: [] for d in DEFINITIONS}

    for row in score_rows:
        for d in DEFINITIONS:
            score_key = f"{d}_score"
            pass_key = f"{d}_pass"
            if score_key in row and pass_key in row:
                s = row[score_key]
                p = row[pass_key]
                def_scores[d].append(float(s) if not isinstance(s, float) else s)
                def_passes[d].append(bool(int(p)) if isinstance(p, str) else bool(p))

    # Pairwise analysis (6 pairs)
    for di, dj in combinations(DEFINITIONS, 2):
        pair_key = f"{di}_{dj}"
        pairwise = compute_pairwise(
            def_scores[di], def_scores[dj], def_passes[di], def_passes[dj]
        )

        # Disagreement characterization
        disagree_rows = []
        for row in score_rows:
            pi = row.get(f"{di}_pass")
            pj = row.get(f"{dj}_pass")
            if pi is None or pj is None:
                continue
            if isinstance(pi, str):
                pi = bool(int(pi))
            if isinstance(pj, str):
                pj = bool(int(pj))
            disagree_rows.append({
                "regime": row.get("regime", ""),
                "family_id": row.get("family_id", 0),
                "Di_pass": bool(pi),
                "Dj_pass": bool(pj),
            })

        pairwise["disagreements"] = characterize_disagreements(disagree_rows)
        result["pairwise"][pair_key] = pairwise

    # Aggregate: Fleiss' kappa (4 raters, 2 categories)
    n_items = len(score_rows)
    if n_items > 0:
        ratings = np.zeros((n_items, 2), dtype=int)
        for i, row in enumerate(score_rows):
            for d in DEFINITIONS:
                pass_key = f"{d}_pass"
                p = row.get(pass_key, False)
                if isinstance(p, str):
                    p = bool(int(p))
                if p:
                    ratings[i, 1] += 1
                else:
                    ratings[i, 0] += 1
        result["aggregate"]["fleiss_kappa"] = fleiss_kappa(ratings)

    # Mean kappa
    kappas = [v["cohens_kappa"] for v in result["pairwise"].values()]
    result["aggregate"]["mean_kappa"] = float(np.mean(kappas)) if kappas else 0.0

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze agreement among D1–D4")
    parser.add_argument("tsv_file", type=Path, help="Score matrix TSV file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    with open(args.tsv_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    result = analyze_agreement(rows)

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output + "\n")


if __name__ == "__main__":
    main()
