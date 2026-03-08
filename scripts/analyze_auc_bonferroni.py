"""Compute Bonferroni-adjusted AUC pairwise confidence intervals.

For the 6 pairwise AUC comparisons (D1 vs D2, D1 vs D3, D1 vs D4,
D2 vs D3, D2 vs D4, D3 vs D4), applies Bonferroni correction:
  adjusted alpha = 0.05 / 6 = 0.0083  →  99.17% CIs

Reads bootstrap_ci.json for raw 95% CIs and bootstrap_ci_strict.json
for strict-mode CIs.

Usage:
    uv run python scripts/analyze_auc_bonferroni.py experiments/benchmark/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFINITIONS = ["D1", "D2", "D3", "D4"]
PAIRS = [(a, b) for i, a in enumerate(DEFINITIONS) for b in DEFINITIONS[i + 1 :]]
N_PAIRS = len(PAIRS)  # 6


def bonferroni_adjusted_ci(lower: float, upper: float, n_comparisons: int) -> tuple[float, float]:
    """Widen a raw 95% CI to Bonferroni-adjusted CI.

    Approximation: if the raw CI corresponds to z=1.96 (95%),
    scale the half-width by z_adjusted / z_raw.
    Adjusted alpha = 0.05 / n_comparisons → use the corresponding z-score.

    For 99.17% CI (α=0.0083): z ≈ 2.638.
    """
    # Compute half-width from original CI (approximated as symmetric)
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0

    # z for original 95% CI
    z_raw = 1.96
    # z for Bonferroni-adjusted CI: scipy.stats.norm.ppf(1 - 0.05 / (2 * n))
    # For n=6: 1 - 0.05/12 = 0.99583 → z ≈ 2.638
    # Pre-computed using scipy.stats.norm.ppf
    z_table = {6: 2.638, 5: 2.576, 4: 2.498, 3: 2.394}
    z_adj = z_table.get(n_comparisons, 2.638)

    scale = z_adj / z_raw
    return center - half_width * scale, center + half_width * scale


def compute_pairwise_diffs(
    auc_data: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Compute pairwise AUC difference CIs.

    Returns dict of "Da_Db" → {diff, lower, upper, bonferroni_lower, bonferroni_upper}.
    Raw CIs are approximated as symmetric around the point estimate difference.
    """
    results = {}
    for a, b in PAIRS:
        ci_a = auc_data[a]["auc_ci"]
        ci_b = auc_data[b]["auc_ci"]
        # Point estimate difference
        center_a = (ci_a[0] + ci_a[1]) / 2.0
        center_b = (ci_b[0] + ci_b[1]) / 2.0
        diff = center_a - center_b

        # Propagate CIs assuming independence (conservative)
        half_a = (ci_a[1] - ci_a[0]) / 2.0
        half_b = (ci_b[1] - ci_b[0]) / 2.0
        half_diff = (half_a**2 + half_b**2) ** 0.5

        raw_lower = diff - half_diff
        raw_upper = diff + half_diff

        bon_lower, bon_upper = bonferroni_adjusted_ci(raw_lower, raw_upper, N_PAIRS)

        key = f"{a}_{b}"
        results[key] = {
            "diff": round(diff, 3),
            "raw_95_ci": [round(raw_lower, 3), round(raw_upper, 3)],
            "bonferroni_99.2_ci": [round(bon_lower, 3), round(bon_upper, 3)],
            "significant_raw": raw_lower > 0 or raw_upper < 0,
            "significant_bonferroni": bon_lower > 0 or bon_upper < 0,
        }
    return results


def main(benchmark_dir: Path) -> None:
    legacy_path = benchmark_dir / "bootstrap_ci.json"
    strict_path = benchmark_dir / "bootstrap_ci_strict.json"

    with legacy_path.open() as f:
        legacy = json.load(f)
    with strict_path.open() as f:
        strict = json.load(f)

    legacy_auc = legacy["roc_auc"]
    strict_auc = strict["roc_auc"]

    print("=" * 60)
    print("AUC Bonferroni-Adjusted Pairwise Comparisons")
    print(f"N pairs = {N_PAIRS}, adjusted alpha = 0.05/{N_PAIRS} = {0.05 / N_PAIRS:.4f}")
    print(f"Adjusted CI level = {(1 - 0.05 / N_PAIRS) * 100:.1f}%  (z ≈ 2.638)")
    print("=" * 60)

    for mode_name, auc_data in [("legacy", legacy_auc), ("strict", strict_auc)]:
        print(f"\n--- {mode_name.upper()} MODE ---")
        print(f"{'Pair':<10} {'Diff':>6}  {'Raw 95% CI':^18}  {'Bonf. 99.2% CI':^20}  Sig?")
        print("-" * 70)
        diffs = compute_pairwise_diffs(auc_data)
        for key, v in diffs.items():
            sig = "Yes" if v["significant_bonferroni"] else "No"
            raw_ci = f"[{v['raw_95_ci'][0]:+.3f}, {v['raw_95_ci'][1]:+.3f}]"
            bon_ci = f"[{v['bonferroni_99.2_ci'][0]:+.3f}, {v['bonferroni_99.2_ci'][1]:+.3f}]"
            print(f"{key:<10} {v['diff']:+6.3f}  {raw_ci:^18}  {bon_ci:^20}  {sig}")

    # Output JSON summary for paper inclusion
    output = {
        "n_pairs": N_PAIRS,
        "adjusted_alpha": round(0.05 / N_PAIRS, 4),
        "adjusted_ci_level_pct": round((1 - 0.05 / N_PAIRS) * 100, 1),
        "z_bonferroni": 2.638,
        "fwer_under_independence": round(1 - 0.95**N_PAIRS, 3),
        "legacy": compute_pairwise_diffs(legacy_auc),
        "strict": compute_pairwise_diffs(strict_auc),
    }
    out_path = benchmark_dir / "auc_bonferroni.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nFWER under independence (raw 95%): {output['fwer_under_independence']:.3f}")
    print(
        "→ Paper note: 'For the 6 pairwise AUC comparisons, family-wise error rate "
        f"under independence would reach ~{output['fwer_under_independence'] * 100:.0f}% "
        f"(1 − 0.95^6); Bonferroni-adjusted {output['adjusted_ci_level_pct']:.1f}% CIs "
        "are reported.'"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_dir>", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]))
