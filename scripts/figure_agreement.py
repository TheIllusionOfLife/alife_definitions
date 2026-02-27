"""Generate 4 publication-quality figures for the agreement analysis.

1. disagreement_heatmap.pdf — Rows: (regime × family); Cols: D1–D4; Color: mean score
2. agreement_matrix.pdf — 4×4: κ (lower triangle), ρ (upper triangle)
3. case_study_timeseries.pdf — 2 panels: F3 (D3 > D2) and F2 (D2 > D3)
4. predictive_roc.pdf — ROC curves per definition on test set

Usage:
    uv run python scripts/figure_agreement.py experiments/benchmark/score_matrix.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFINITIONS = ["D1", "D2", "D3", "D4"]
FIGURE_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


def _setup_matplotlib():
    """Configure matplotlib for publication figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    return plt


# ---------------------------------------------------------------------------
# Figure 1: Disagreement Heatmap
# ---------------------------------------------------------------------------


def figure_disagreement_heatmap(rows: list[dict], out_path: Path) -> None:
    """Rows: (regime × family); Cols: D1–D4; Color: mean score."""
    plt = _setup_matplotlib()

    # Group by (regime, family_id) → mean score per definition
    groups: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        regime = row["regime"]
        fid = int(row["family_id"])
        key = (regime, fid)
        if key not in groups:
            groups[key] = {d: [] for d in DEFINITIONS}
        for d in DEFINITIONS:
            groups[key][d].append(float(row[f"{d}_score"]))

    # Sort keys: by regime then family_id
    keys = sorted(groups.keys())
    if not keys:
        print("SKIP: no data for disagreement heatmap")
        return

    labels = [f"{r}/F{f + 1}" for r, f in keys]
    matrix = np.array([[np.mean(groups[k][d]) for d in DEFINITIONS] for k in keys])

    fig, ax = plt.subplots(figsize=(4.5, max(2.5, len(keys) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(DEFINITIONS)))
    ax.set_xticklabels(DEFINITIONS)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Definition")
    ax.set_ylabel("Regime / Family")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Score", shrink=0.8)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Agreement Matrix
# ---------------------------------------------------------------------------


def figure_agreement_matrix(rows: list[dict], out_path: Path) -> None:
    """4×4 matrix: κ (lower triangle), ρ (upper triangle)."""
    plt = _setup_matplotlib()
    from analyze_agreement import cohens_kappa
    from scipy import stats

    n = len(DEFINITIONS)
    kappa_mat = np.full((n, n), np.nan)
    rho_mat = np.full((n, n), np.nan)

    for i, di in enumerate(DEFINITIONS):
        for j, dj in enumerate(DEFINITIONS):
            if i == j:
                continue
            passes_i = [
                bool(int(r[f"{di}_pass"]))
                if isinstance(r[f"{di}_pass"], str)
                else bool(r[f"{di}_pass"])
                for r in rows
            ]
            passes_j = [
                bool(int(r[f"{dj}_pass"]))
                if isinstance(r[f"{dj}_pass"], str)
                else bool(r[f"{dj}_pass"])
                for r in rows
            ]
            scores_i = [float(r[f"{di}_score"]) for r in rows]
            scores_j = [float(r[f"{dj}_score"]) for r in rows]

            kappa_mat[i, j] = cohens_kappa(passes_i, passes_j)
            if np.std(scores_i) > 0 and np.std(scores_j) > 0:
                rho, _ = stats.spearmanr(scores_i, scores_j)
                rho_mat[i, j] = rho if not np.isnan(rho) else 0.0
            else:
                rho_mat[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(4, 3.5))
    # Display combined matrix
    display = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                display[i, j] = 1.0
            elif i > j:
                display[i, j] = kappa_mat[i, j]  # lower: kappa
            else:
                display[i, j] = rho_mat[i, j]  # upper: rho

    im = ax.imshow(display, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(DEFINITIONS)
    ax.set_yticks(range(n))
    ax.set_yticklabels(DEFINITIONS)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8)
            elif i > j:
                val = kappa_mat[i, j]
                label = f"k={val:.2f}" if not np.isnan(val) else "n/a"
                ax.text(j, i, label, ha="center", va="center", fontsize=7)
            else:
                val = rho_mat[i, j]
                label = f"r={val:.2f}" if not np.isnan(val) else "n/a"
                ax.text(j, i, label, ha="center", va="center", fontsize=7)

    ax.set_title("Lower: Cohen's k | Upper: Spearman r", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Case Study Time Series
# ---------------------------------------------------------------------------


def figure_case_study(run_summary: dict | None, out_path: Path) -> None:
    """2 panels: F3 (D3 > D2) and F2 (D2 > D3)."""
    if run_summary is None:
        print("SKIP: no run data for case study figure")
        return

    plt = _setup_matplotlib()
    from adapters.common import discover_family_ids, extract_family_series

    available_fids = set(discover_family_ids(run_summary))
    if not ({1, 2} <= available_fids):
        print("SKIP: case study requires family IDs 1 and 2")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)

    # F3 (family_id=2): passes D3 / fails D2
    f3 = extract_family_series(run_summary, 2)
    ax1.plot(f3["alive_count"], label="alive", linewidth=1)
    ax1.plot(f3["energy_mean"] * 10, label="energy (×10)", linewidth=0.8, alpha=0.7)
    ax1.plot(f3["boundary_mean"] * 10, label="boundary (×10)", linewidth=0.8, alpha=0.7)
    ax1.set_title("F3: autonomy (D3 > D2)", fontsize=9)
    ax1.set_xlabel("Sample step")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=6)

    # F2 (family_id=1): passes D2 / lower D3
    f2 = extract_family_series(run_summary, 1)
    ax2.plot(f2["alive_count"], label="alive", linewidth=1)
    ax2.plot(f2["birth_count"], label="births", linewidth=0.8, alpha=0.7)
    ax2.plot(f2["genome_diversity"] * 10, label="diversity (×10)", linewidth=0.8, alpha=0.7)
    ax2.set_title("F2: Darwinian (D2 > D3)", fontsize=9)
    ax2.set_xlabel("Sample step")
    ax2.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Predictive ROC curves
# ---------------------------------------------------------------------------


def figure_predictive_roc(predictive_json: dict | None, out_path: Path) -> None:
    """ROC curves per definition on test set."""
    if predictive_json is None:
        print("SKIP: no predictive data for ROC figure")
        return

    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5, label="Random")

    defn_data = predictive_json.get("definitions", {})
    if not defn_data:
        print("SKIP: no definition data in predictive JSON")
        plt.close(fig)
        return

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for (defn, metrics), color in zip(defn_data.items(), colors, strict=False):
        auc_val = metrics.get("roc_auc", float("nan"))
        label = f"{defn} (AUC={auc_val:.2f})" if not np.isnan(auc_val) else f"{defn} (n/a)"
        # Plot a point at (1-specificity, sensitivity) from the frozen threshold
        ba = metrics.get("balanced_accuracy", 0.5)
        recall = metrics.get("recall", 0.5)
        ax.scatter([1 - (2 * ba - recall)], [recall], color=color, s=40, zorder=5)
        ax.annotate(
            label,
            xy=(0.05, 0.95 - 0.08 * DEFINITIONS.index(defn)),
            xycoords="axes fraction",
            fontsize=7,
            color=color,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Predictive Validity (ROC)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: generate publication figures from score matrix."""
    parser = argparse.ArgumentParser(description="Generate agreement figures")
    parser.add_argument("tsv_file", type=Path, help="Score matrix TSV")
    parser.add_argument("--case-study-json", type=Path, help="Single run JSON for case study")
    parser.add_argument("--predictive-json", type=Path, help="Predictive analysis JSON")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load score matrix
    if not args.tsv_file.exists():
        print(f"SKIP: {args.tsv_file} not found")
        return
    with open(args.tsv_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("SKIP: empty score matrix")
        return

    figure_disagreement_heatmap(rows, FIGURE_DIR / "disagreement_heatmap.pdf")
    figure_agreement_matrix(rows, FIGURE_DIR / "agreement_matrix.pdf")

    # Case study
    case_run = None
    if args.case_study_json and args.case_study_json.exists():
        with open(args.case_study_json) as f:
            case_run = json.load(f)
    figure_case_study(case_run, FIGURE_DIR / "case_study_timeseries.pdf")

    # Predictive ROC
    pred_data = None
    if args.predictive_json and args.predictive_json.exists():
        with open(args.predictive_json) as f:
            pred_data = json.load(f)
    figure_predictive_roc(pred_data, FIGURE_DIR / "predictive_roc.pdf")


if __name__ == "__main__":
    main()
