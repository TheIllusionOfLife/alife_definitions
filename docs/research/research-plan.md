# Research Plan (Markdown) — “Benchmarking Multiple Definitions of Life in a Shared Digital World” (for ALIFE Full Paper)

This document proposes a **drastic extension** of your current “7 textbook criteria + functional analogy + ablation” framework into a **comparative benchmark across multiple definitions of life**, executed in the *same* digital world and (optionally) with coexisting agent types. It is designed to fit an **ALIFE conference full paper** narrative and deliver memorable, reusable artifacts beyond incremental polishing of the current work. 

---

## 0) One-sentence ambition (what reviewers will remember)

> **Different definitions of life disagree systematically—even on the same organisms in the same world—and we provide an empirical benchmark (datasets + adapters + metrics) that makes those disagreements measurable, reproducible, and scientifically useful.**

---

## 1) Why this is a “drastic upgrade” over the current paper

Your current paper makes a strong claim: you can implement the **7 textbook criteria** as **interdependent processes** and show necessity via ablation, with clean stats and held-out seeds. 
The limitation is not rigor—it’s *scope of the question*: you validate one operational set.

This new direction shifts the question to:

1. **Definitions-as-measurement-regimes:** life is not one binary label.
2. **Empirical disagreement:** show *where* and *why* definitions disagree.
3. **Benchmark artifact:** provide reusable “definition adapters” so other ALife systems can be evaluated.

This produces a paper that feels like a **field contribution**, not “one more improved organism.”

---

## 2) Key research questions (RQ)

### RQ1 — Definition disagreement

* When applied to the **same** digital organisms/worlds, **which definitions label what as “life,” and where do they disagree?**

### RQ2 — Disagreement structure

* Are disagreements random noise, or do they have **structure** (e.g., autonomy-based definitions accept some cases that Darwinian definitions reject)?

### RQ3 — Robustness under perturbation

* Under controlled perturbations (resource scarcity, toxins/waste pressure, sensing noise), **which definitions track robustness/persistence better**?

### RQ4 — Transferability

* Can a definition adapter be applied to **other substrates** (optional stretch goal), or at least remain stable across varied environments within your substrate?

---

## 3) Core contribution package (what you will ship)

### 3.1 The “Life Definitions Benchmark” (LDB)

A dataset + evaluation suite including:

* **World variants** (environment regimes)
* **Organism variants** (your system + optional additional agent families)
* **Definition adapters** (scoring functions + necessary processes + ablations)
* **Agreement/disagreement analysis** tooling

### 3.2 Definition adapters (the centerpiece)

Each definition becomes:

* **Operational criteria** (measurable signals)
* **Decision rule** (score / threshold / multi-criteria)
* **Ablation mapping** (if applicable)

Your existing “functional analogy” framework already has a clean ablation hook model. 

---

## 4) Which “definitions of life” to include (recommended set)

You want **3–5** definitions: enough to disagree meaningfully, not so many you drown.

### Definition D1 — Textbook 7-criteria (baseline)

* This is your current operational bridge. 
* Adapter: existing criterion toggles + your “dynamic + degradation + feedback coupling” test.

### Definition D2 — Darwinian / NASA-style (life as evolving replicators)

* Minimal lens: **sustained self-reproduction + heritable variation + differential success**
* Adapter: focus on reproduction + evolution metrics, with heredity and selection evidence.

### Definition D3 — Autonomy / Organizational closure (life as self-maintaining organization)

* Lens: **mutual dependence** among processes supporting persistence (closure).
* Adapter: build a closure score from intervention effects / causal influence among process variables (you already record energy, waste, boundary, internal state, etc.). 

### Definition D4 — Process / Information maintenance (life as persistence of organized information)

* Lens: not just having a genome, but **information that causally maintains itself**
* Adapter: measure information retention across generations + functional contribution (e.g., genome → phenotype → survival).

### Definition D5 (optional) — “Minimal metabolism-first” (life as far-from-equilibrium maintenance proxy)

* Only if you can define a clean proxy without overclaiming thermodynamics:

  * sustained resource uptake + waste output + internal regulation against decay

> Recommendation: start with **D1–D4**; D5 is optional.

---

## 5) Experimental design overview

### 5.1 Two modes: choose one primary, keep the other as optional

**Mode A (recommended for clarity):**
Run the *same organism population* and apply multiple definition adapters to the resulting traces.

**Mode B (more ambitious, but heavier):**
Spawn multiple agent families in the same world (e.g., your organisms + “Darwinian minimal replicators” + “autonomy-heavy non-replicators”), then compare definitions in a mixed ecology.

For ALIFE, Mode A already yields a strong paper; Mode B adds showmanship but increases engineering risk.

---

## 6) What data to collect (instrumentation plan)

You already have state variables and manifests. 
For definition comparisons, add a standardized “trace schema”:

### 6.1 Per-organism time series (sampled every K steps)

* Boundary integrity `b`
* Energy `e`, resource `r`, waste `w`
* Internal homeostasis state `s` (or summary stats)
* Movement / sensing outcomes (distance traveled, resource encounter rate)
* Development stage / maturity `m`
* Alive/dead event time

### 6.2 Per-lineage / reproduction events

* Parent ID, child ID, generation
* Genome snapshot hash (or full vector if affordable)
* Mutation events summary (counts, magnitudes)

### 6.3 Per-population summary

* Alive count trajectory, AUC
* Diversity metrics (genome diversity; you already compute something similar) 
* Spatial cohesion (you already use mean pairwise distance) 

### 6.4 Perturbation metadata

* Which environment regime (see §7)
* Perturbation timing (step ranges)

---

## 7) Environment regimes (make definitions disagree on purpose)

To reveal disagreement, you need regimes that stress different “philosophies of life”:

### Regime E1 — Baseline (your current stable setup)

* For calibration and sanity.

### Regime E2 — Resource shock

* Periodic resource scarcity spikes.
* Tests autonomy/homeostasis vs pure Darwinian replication.

### Regime E3 — Waste/toxin pressure

* Increase waste penalty (boundary decay coupling) and/or waste accumulation.
* Tests metabolism + regulation significance.

### Regime E4 — Sensing noise / impaired perception

* Degrade inputs to the controller.
* Tests responsiveness and the role of agency/behavior.

### Regime E5 — Spatial patchiness

* Clustered resources vs uniform.
* Distinguishes adaptive movement and organization.

Keep the number small (3–5) but conceptually distinct.

---

## 8) Definition adapters: operationalization details

Below is a concrete “adapter template” and recommended operationalizations for each definition.

### 8.1 Adapter template

Each adapter should specify:

1. **Observable signals** (from trace schema)
2. **Score function** `S ∈ [0, 1]` or multi-dimensional score
3. **Decision rule** (threshold or Pareto)
4. **Failure modes** it is sensitive to
5. **Ablation expectations** (optional)

---

### 8.2 D1 adapter — Textbook 7-criteria + functional analogy

Reuse your existing operationalization:

* Dynamic process
* Measurable degradation upon ablation
* Feedback coupling 

**Decision rule example:**

* Life(D1) if all 7 criteria satisfy the 3 functional-analogy conditions (or a weighted score if you want graded life-likeness).

**Note:** For cross-definition comparison, consider producing both:

* a **binary** label (passes D1 / fails D1)
* a **graded** score (how strongly each criterion passes)

---

### 8.3 D2 adapter — Darwinian / NASA-style (evolving replicators)

**Signals**

* Sustained reproduction events over time
* Heritability proxy: parent-offspring genome similarity
* Differential success: lineage growth correlated with traits

**Score ideas**

* `S_reprod = normalized births per unit time under steady state`
* `S_hered = corr(parent_genome, child_genome) - baseline`
* `S_select = positive association between trait and survival/reproduction`

**Decision rule**

* Life(D2) if reproduction persists AND heredity exists AND there is nonzero selection signal.

---

### 8.4 D3 adapter — Autonomy / closure

This is the most “drastic” and will make the paper stand out.

**Signals**

* Process variables: boundary, energy, waste, internal state, movement effectiveness
* Intervention effects (you already compute cross-criterion effects) 

**Closure score (practical version)**

* Build a directed influence graph between process variables using:

  * intervention effects (best)
  * lagged predictability (secondary)
* Define closure as:

  * size/strength of the **largest strongly connected component** among “maintenance processes”
  * normalized by total processes

**Decision rule**

* Life(D3) if closure score exceeds threshold AND system maintains itself under at least one perturbation regime (persistence requirement).

> This avoids “closure as philosophy” and turns it into a measurable graph property.

---

### 8.5 D4 adapter — Information maintenance / causal information

**Signals**

* Genome diversity over time
* Functional relevance: genome differences predict survival/reproduction

**Score idea**

* `S_info = mutual_information(genome_features, fitness_proxy)` (approx via regression / predictability)
* Track whether information is **preserved and used**, not just present.

**Decision rule**

* Life(D4) if there exists stable heritable information that causally predicts persistence metrics beyond noise.

---

## 9) Analysis: how to compare definitions

### 9.1 Agreement & disagreement

Produce:

* Pairwise agreement matrix between definitions (Cohen’s κ for binary labels; rank correlation for scores)
* Venn-style overlaps (counts)

### 9.2 “Where they disagree”

For organisms/worlds where D_i ≠ D_j:

* characterize them by:

  * perturbation regime
  * process profiles (energy stability, boundary integrity, reproduction rate)
  * closure score

This yields an interpretable statement like:

> “Autonomy-based definitions label X as life because it maintains closure under shocks, even when Darwinian definitions reject it due to weak evolution.”

### 9.3 Predictive validity (important)

Not “which definition is true,” but:

* Which definition’s score predicts **robust persistence** under unseen perturbations?

Concretely:

* Train thresholds on calibration seeds/regimes
* Evaluate on held-out seeds/regimes (you already separate calibration vs test seeds; keep that standard) 

---

## 10) Figures that will make the ALIFE paper pop

1. **Definition disagreement heatmap**
   Rows: organisms/runs; columns: definitions; color: life score.

2. **Agreement matrix**
   With κ / correlations.

3. **Phase diagram** (optional but powerful)
   x-axis: perturbation harshness; y-axis: resource availability
   Color: which definitions label “life” regions.

4. **Case studies (2–3)**
   Show one organism/run that:

   * passes D3 but fails D2
   * passes D2 but fails D3
     Include time series plots (boundary/energy/reproduction) and one short narrative.

---

## 11) Paper outline (ALIFE Full Paper)

### Title (suggestion)

**“Life Definitions Disagree: An Empirical Benchmark of Competing Operationalizations in a Shared Digital Ecology”**

### Abstract

* 1–2 sentences: problem (definitions differ)
* Contribution: benchmark + adapters + disagreement findings
* Key results: disagreement structure + which scores predict robustness

### 1. Introduction

* Why definitions matter (life detection, ALife foundations)
* Gap: most work adopts one definition
* Your approach: empirical comparison in the same world

### 2. Related work

* Definitions of life (brief, functional)
* Benchmarking philosophies (not deep philosophy—keep it ALife)

### 3. System & baseline (short)

* Summarize your current system as substrate
* Cite that the 7-criteria integration exists and is validated 

### 4. Definition adapters

* D1–D4 operationalizations
* Trace schema

### 5. Experiments

* regimes E1–E4
* held-out seeds
* metrics and stats

### 6. Results

* disagreement heatmaps
* agreement matrix
* predictive validity under perturbations
* case studies

### 7. Discussion

* what disagreements imply
* limits: adapter dependence, substrate bias
* how others can plug in their systems

### 8. Artifact release

* benchmark suite structure + minimal reproduction instructions

---

## 12) Practical recommendations to keep scope sane

* **Don’t build new organisms first.** Start by applying multiple definitions to the *same* traces (Mode A).
* Keep definitions to **4** (D1–D4).
* Keep environments to **3–4** regimes.
* Make the closure adapter “good enough” using intervention effects already in your pipeline. 
* Add 2–3 memorable case studies.

---

## 13) Questions for you (to update this doc)

Please answer these so I can refine the plan into a more executable spec (including exact metrics, thresholds, and which plots to prioritize):

1. **Which non-textbook definitions do you most want to include?**

   * Do you already have a shortlist (e.g., NASA/Darwinian, autopoiesis/closure, chemoton-inspired, etc.)?

2. **Do you want “life” to be binary per definition, or a graded score?**

   * My recommendation: produce both (score + thresholded label), but choose one as the headline.

3. **Do you prefer Mode A or Mode B?**

   * Mode A: same population, multiple labels (safer)
   * Mode B: mixed ecologies with distinct agent families (flashier)

4. **What perturbation regimes are easiest to implement in your simulator right now?**

   * Resource shock, waste/toxin increase, sensing noise, spatial patchiness—any constraints?

5. **What’s your target submission type and page limit for ALIFE?**

   * (Some venues have strict figure limits; this affects how many regimes/definitions we can present.)

6. **Do you have a preferred “north star” application framing?**

   * e.g., “life detection in astrobiology,” “deep biosphere,” or “AI agents as semi-life systems.”

Reply with your answers (even short bullet points), and I’ll update this markdown into a **ready-to-execute research protocol** with:

* exact adapter formulas (as pseudocode),
* exact statistical tests and reporting format,
* a figure-by-figure checklist.
