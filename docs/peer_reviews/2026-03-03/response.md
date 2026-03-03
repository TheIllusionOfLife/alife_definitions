# Response to Peer Reviews — ALIFE 2026

**Paper**: "Life Definitions Disagree: An Empirical Benchmark of Competing Operationalizations in a Shared Digital Ecology"

**Date**: 2026-03-03

We thank all three reviewers for their thorough and constructive feedback. Below we address each concern point-by-point, referencing the specific changes made in the revised manuscript.

---

## Response to Review A (6.5/10, Weak Accept)

### Major Concerns

**A§4.1 — Operationalization arbitrariness and TE/Granger stability**

> "操作化自体が結果を大きく左右します" / TE/Granger estimation details are unclear

We agree that operationalization choices matter and have addressed this on multiple fronts:

1. **TE/Granger estimation details** are now fully specified in §4 (D3 adapter): 5 equal-width bins on z-scored series, lag 1, 400 time-shifted permutations for significance; Granger causality uses OLS at lags 1–5 with Holm–Bonferroni correction. Sample size is n=200 (10-step intervals over 2,000 steps).

2. **Estimator robustness sweep** (new analysis `analyze_te_robustness.py`): We sweep bins ∈ {5, 10, 20} × lag ∈ {1, 2, 3} on calibration seeds and report Spearman rank correlation of D3 closure scores across settings. Results confirm rank stability.

3. **Surrogate FPR expanded to all regimes** (§6): Previously E1 only; now reports per-regime and cross-regime mean/max FPR across E1–E5.

4. **Alternative operationalizations** for each definition (new §5, exploratory analysis):
   - D1: geometric/arithmetic/harmonic/min aggregation modes
   - D3: Bonferroni (default) vs AND-condition edge detection (requiring both TE and Granger significant)
   - D4: hash-based vs L2-distance genome similarity
   - All reported as exploratory robustness checks with rank stability analysis

**A§4.2 — D2 selection indicator is weak**

> Price equation or more direct fitness decomposition should be considered

This was addressed in a prior revision. D2's selection sub-score now uses the **Price equation** (§4.2): $\Delta\bar{z} = \text{Cov}(w, z)/\bar{w}$, where $w$ is per-parent offspring count and $z$ is mean parent–child genome distance. This provides a direct fitness decomposition from lineage records, replacing the earlier correlation-based proxy. We now additionally emphasize in the Results that this decomposition avoids conflating selection with drift (§6, Price equation paragraph).

The transmission component is acknowledged as a limitation (§7, Limitation 3), with a more specific discussion of why it requires multi-generation lineage tracking.

**A§4.3 — Competition confounds**

> Single-family controls are needed to separate capability deficits from competition effects

Single-family controls (3 families × 5 regimes × 200 seeds) were added in a prior revision and are reported in §5 and §6. Key finding: the D2–D3 disagreement axis is preserved in isolation, confirming that definition disagreements reflect capability differences, not competition artifacts. F3's D3 closure score improves by +0.16 without competition (§6, Competition controls).

**A§4.4 — D3 closure × persistence circularity**

> D3 score includes population persistence, creating circularity with alive-count AUC target

Addressed in a prior revision: the primary D3 score uses **closure_only mode** (fraction of process variables in the largest SCC), which does not include population persistence. The composite (closure × persistence) is reported separately and is not used in predictive validity evaluation (§4.3).

**A§4.5 — Quantitative results insufficient (placeholder figures)**

> Figures are placeholders; numerical results missing

All figures now contain real data: disagreement heatmap (Fig. 2), agreement matrix (Fig. 3), case study time series (Fig. 4), ROC curves (Fig. 5). All numerical results (κ, ρ, AUC, CIs) are reported inline with LaTeX macros sourced from the analysis pipeline.

**A§4.6 — Double-blind compliance**

> Identifying information may be present

The manuscript uses "Anonymous" as author, defers repository URL to acceptance, and references Zenodo DOI as "to be assigned upon acceptance" (§8).

### Minor Comments

**A§5.1** — D4's relationship to modern information-theoretic views: Walker & Davies (2013) is now cited and discussed in §2 and §4.4.

**A§5.2** — D1 criterion-to-variable mapping: Table 1 now provides the complete mapping.

**A§5.3** — Default threshold rationale and calibrated values: Table 2 now reports both default (0.3) and calibrated thresholds per definition.

**A§5.4** — Multiple comparison corrections: A new paragraph in §5 (Multiple comparison corrections) specifies the correction hierarchy: per-pair Bonferroni + BH-FDR for D3 edges, descriptive bootstrap CIs for agreement metrics, raw 95% CIs for pairwise AUC differences with family-wise caveat.

**A§5.5** — A priori predictions for regimes: Now stated explicitly in §5 (Environment regimes paragraph).

### Questions for Authors

**A§Q1** (TE/Granger details): Fully specified in §4.3; robustness sweep in supplementary analysis. See response to A§4.1 above.

**A§Q2** (D2 selection indicator): Price equation implemented. See response to A§4.2 above.

**A§Q3** (Single-family runs): Yes, added. See response to A§4.3 above.

**A§Q4** (D3 closure-only rationale): Closure-only is primary; composite reported separately. See response to A§4.4 above.

**A§Q5** (Unseen conditions = unseen seeds or regimes?): Clarified in §5: "unseen conditions refers to unseen seeds within the same regimes, not unseen regimes." Additionally, **Leave-One-Regime-Out (LORO) cross-validation** has been implemented (new `analyze_loro.py`): 5-fold evaluation holding out one regime, calibrating on the other four, and evaluating on the held-out regime's test seeds. This addresses the cross-regime generalization question directly.

### Improvement Suggestions (§11)

**A§11.1** (Cross-substrate): Lenia cross-substrate prototype implemented (§6.1, new `scripts/lenia/` package). Pre-registered hypothesis H_Lenia tests whether D3 > D2 in a non-reproductive substrate.

**A§11.2** (Alternative operationalizations): Implemented for D1/D3/D4 with rank stability analysis. See response to A§4.1.

**A§11.3** (Leave-one-regime-out): Implemented. See response to A§Q5.

**A§11.4** (TE/Granger estimator sweep): Implemented. See response to A§4.1.

**A§11.5** (Price transmission): Acknowledged as limitation with expanded discussion. Multi-generation tracking is left for future work.

**A§11.6** (Additional family types): Beyond current scope; noted as future work in §7 (Extensibility).

**A§11.7** (Pareto predictive diagram): Implemented in `figure_agreement.py` — 2D scatter of alive-AUC vs recovery-AUC with lineage-diversity as marker size.

**A§11.8** (Reproducibility): Computation cost table added (Table 3). Version pinning paragraph added (§8). Full reproduce_all.sh pipeline maintained.

---

## Response to Review B (7.8/10, Weak Accept)

### Major Concerns

**B§4.1 — Paired bootstrap difference CIs**

> "D1がD4より有意に高いか" — paired bootstrap difference CIs would strengthen conclusions

Implemented in `analyze_bootstrap_ci.py`: `bootstrap_pairwise_differences()` computes AUC(Di) − AUC(Dj) for all 6 definition pairs within each of 2,000 bootstrap iterations, reporting 95% CIs. Significance is declared when the interval excludes zero.

**B§4.2 — D1 β dependence on family composition**

> D1 scores change with co-existing family composition

We added `sweep_d1_beta_reference()` in `analyze_sensitivity.py`, testing D1 with different reference family subsets ({F2 only}, {F3 only}, {no β}). Rank correlation vs default is reported. Additionally, the Discussion (§7) now explicitly discusses this as a property inherent to comparative definitions, and the Threats to Validity table (Table 4) lists it with the mitigation.

**B§4.3 — Statistical comparison insufficiency / multiple comparisons**

> Multiple comparison correction strategy should be explicit

New paragraph in §5 (Multiple comparison corrections) details the full hierarchy. See response to A§5.4.

**B§4.4 — Surrogate FPR scope**

> Surrogate FPR is E1-only; extend to all regimes

Implemented: `analyze_surrogate_fpr.py` now supports `--regimes E1,E2,E3,E4,E5` and reports per-regime + cross-regime mean/max FPR.

**B§4.5 — External validity (cross-substrate)**

> Testing on different ALife substrates would increase impact

Lenia cross-substrate prototype implemented (§6.1). See response to A§11.1.

### Questions for Authors

**B§Q1** (D1 β rank stability with different reference families): Implemented as `sweep_d1_beta_reference()`. See response to B§4.2.

**B§Q2** (Binarization threshold alternatives): `_make_labels()` in `analyze_predictive.py` now accepts `binarization` parameter: "median" (default), "q25", "q75". Results reported as exploratory robustness check.

**B§Q3** (D3 AND-condition edge detection): Implemented as `edge_mode="and"` in `score_d3()`. Under AND-condition, both TE and Granger must be significant (intersection-union test using max(p_TE, p_Granger)). Results reported via `sweep_d3_edge_mode()` in sensitivity analysis.

**B§Q4** (D4 causal component weight rationale): Now explained in §4.4: "The 2× causal weight reflects D4's core thesis that information must do causal work—mere presence or persistence is necessary but not sufficient."

### Improvement Suggestions (§7)

**B§A.1** (Paired bootstrap differences): Implemented. See B§4.1.

**B§A.3** (Multiple comparison corrections): Explicit hierarchy documented. See A§5.4.

**B§A.6** (Leave-one-regime-out): Implemented. See A§Q5.

**B§A.7** (D1 β sensitivity): Implemented. See B§4.2.

**B§A.8–A.10** (Artifact reproducibility, traceability, pre-registration): Compute cost table (Table 3), version pinning paragraph, and confirmatory vs exploratory declaration all added.

**B§A.11** (Claim scoping): Predictive validity trade-offs paragraph in §7 now consistently uses "in this substrate and for this target" qualifiers.

**B§A.12** (Definition selection flowchart): Noted for camera-ready; the extensibility paragraph in §7 provides guidance on when each definition is most appropriate.

---

## Response to Review C (Accept)

We are grateful for the positive assessment and the detailed improvement suggestions. We address all remaining concerns below.

### Remaining Concerns

**C§R1 — D3 SCC on 5-node graph**

> SCC of 1.6/5 is small; interpretation is scale-dependent

We acknowledge this is a limitation of the current 5-process-variable design. The Threats to Validity table notes this implicitly under "TE/Granger estimation noise." The choice of 5 variables reflects the core processes observable in the simulation (energy, waste, boundary, birth count, maturity). Expanding the variable set would require additional simulation instrumentation; we note this as future work.

**C§R2 — Prediction target inter-correlation**

> Correlation between alive-AUC, recovery time, and lineage diversity is not reported

Implemented: `compute_target_correlations()` in `analyze_predictive.py` computes pairwise Spearman correlations between all three targets. Results are reported in the CLI output and referenced in the Results section.

**C§R3 — Price equation transmission component**

> Limitation (3) should discuss more concretely why transmission is missing

Expanded in §7 Limitations: "the full transmission–selection decomposition (requiring multi-generation lineage tracking with parent–offspring trait regression) is left for future work."

**C§R4 — E4 sensing noise impact on D2**

> E4/F2 D2 score drop deserves discussion beyond D1's γ condition

Added to §7 (Regime dependence): "Specifically, E4's Gaussian noise (σ = 0.5) on neural inputs degrades resource harvesting efficiency, reducing the energy–waste coupling signal that D1's metabolism criterion relies on; D2's reproduction and selection sub-scores, computed from discrete lineage events, are comparatively robust to continuous sensory perturbation."

**C§R5 — κ vs ρ asymmetry explanation**

> D1–D3: ρ > κ reversal pattern deserves explanation

Added to §6 (Agreement matrix paragraph): "The κ–ρ asymmetry (e.g., D1–D3: κ = 0.48 vs ρ = 0.78) arises because κ measures agreement at a fixed binary threshold, while ρ captures monotonic rank correspondence across the graded score range; definitions can rank families similarly while disagreeing on which cross the pass/fail boundary."

### Minor Issues

**C§M1** (Mode A/B naming): Mode A now labeled "(single-population)" and Mode B labeled "(multi-family)" in §3.

**C§M2** (Genome hash quantization): Added to §4.4: "continuous-valued floats are hashed directly without rounding, so any mutation (however small) produces a distinct hash with high probability." Additionally, L2-distance genome similarity is now available as an alternative operationalization for D4.

**C§M3** (Version pinning): New paragraph in §8 references `uv.lock` and `rust-toolchain.toml`.

**C§M4** (Cross-reference placeholders): Section references now use `\S\ref{sec:...}` throughout.

### Questions for Authors

**C§Q1** (SCC temporal stability): Implemented as `analyze_temporal_d3.py` — splits runs into time windows and computes Jaccard similarity of SCC membership between windows. This directly quantifies whether closure is transient or persistent. Results are discussed in §7 Limitations (item 4).

**C§Q2** (5th definition candidate): We agree that an agency-based definition would add a third independent axis. This is noted as future work in §7 (Extensibility).

**C§Q3** (JSON schema mapping to other substrates): The Lenia cross-substrate prototype (§6.1) demonstrates exactly this: a priori mapping from Lenia variables (mass → energy_mean, compactness → boundary_mean, entropy → waste_mean, components → alive_count) to the existing run_summary JSON schema. No adapter code changes were required.

### Major Improvement Suggestions

**C§P1** (Cross-substrate): Lenia prototype implemented. See A§11.1.

**C§P2** (Alternative operationalizations): Implemented for D1/D3/D4. See A§4.1.

**C§P3** (5th definition): Noted as future work. See C§Q2.

**C§P4** (Writing structure): Results narrative summary added at §6 opening. κ vs ρ asymmetry explained. Compute cost consolidated into table. Confirmatory vs exploratory declaration added.

**C§P5** (Temporal dynamics): Time-window D3 analysis implemented. See C§Q1.

**C§P6** (Community leaderboard): Beyond paper scope; noted for post-publication development.

---

## Summary of Changes

| Phase | Description | Key Files |
|-------|-------------|-----------|
| 0 | Performance optimization (parallelization, TE cache) | `experiment_benchmark.py`, `cache.py` |
| 1 | Statistical rigor (paired CIs, all-regime FPR, TE sweep, β stability) | `analyze_bootstrap_ci.py`, `analyze_surrogate_fpr.py`, `analyze_te_robustness.py`, `analyze_sensitivity.py` |
| 2 | Alternative operationalizations (D1 aggregation, D3 edge mode, D4 similarity) | `adapters/d1.py`, `adapters/d3.py`, `adapters/d4.py` |
| 3 | Robustness figure + predictive enhancements | `figure_agreement.py`, `analyze_predictive.py` |
| 4 | Leave-One-Regime-Out cross-validation | `analyze_loro.py` |
| 5 | Temporal D3 SCC stability | `analyze_temporal_d3.py` |
| 6 | Lenia cross-substrate prototype | `scripts/lenia/` |
| 7 | Paper text revisions | `paper/main.tex` |

All 246 existing tests continue to pass. The paper compiles without errors.
