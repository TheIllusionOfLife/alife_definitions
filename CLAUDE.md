# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ALife Definitions** is a research project benchmarking multiple operational definitions of life in the same digital world. The Rust+Python simulation substrate (inherited from the digital-life-v1 project) is used to empirically compare how competing definitions agree and disagree across organisms and environment regimes.

**Target venue**: ALIFE 2026 Full Paper (8p), deadline ~April 1, 2026.

**Stance**: Weak ALife — the system is a functional model, not a claim that the organisms are "alive."

## Document Structure

| Document | Role |
|----------|------|
| `docs/research/research-plan.md` | **Authoritative plan**: new benchmark direction, definitions D1–D4, experimental design, adapter templates |

`docs/archive/digital-life-v1/` contains the prior project's research artifacts (action-plan, unified-review, paper) — read-only historical reference.

## Architecture Decisions

- **Hybrid two-layer substrate**: Swarm agents (10-50 per organism) form organism-level structures; organisms (10-50) inhabit a continuous 2D environment
- **Language**: Rust (core simulation) + Python (experiment management, analysis, definition adapters). Bound via PyO3/maturin
- **Build**: `uv` for Python, `cargo` + `maturin develop` for Rust extension
- **LaTeX**: Use `tectonic` for paper compilation (not pdflatex/latexmk)
- **Compute**: Mac Mini M2 Pro. Target: >100 timesteps/sec for 2,500 agents

## Benchmark Definitions

The central contribution is empirically comparing these four operational definitions on the same organism traces:

- **D1** — Textbook 7-criteria (cellular org, metabolism, homeostasis, growth, reproduction, stimuli response, evolution) + functional-analogy test
- **D2** — Darwinian/NASA (sustained self-reproduction + heritable variation + differential success)
- **D3** — Autonomy/organizational closure (mutual process dependence; closure score from intervention effects)
- **D4** — Information maintenance (genome causally predicts survival/reproduction; information is preserved and used)

## Experimental Design

**Primary mode (Mode A)**: Run the same organism population; apply multiple definition adapters to the resulting traces.

**Environment regimes**: E1 (baseline), E2 (resource shock), E3 (waste/toxin pressure), E4 (sensing noise). 3–5 regimes total.

**Data separation protocol**:
- Calibration set: seeds 0-99 (threshold tuning, adapter development)
- Final test set: seeds 100-199 (held-out evaluation with fixed thresholds)
- Statistics: Mann-Whitney U, Holm-Bonferroni correction, Cohen's d

## Python Package

- Import name: `alife_defs`
- Project name: `alife-defs`
- Rust crates: `alife-defs-core`, `alife-defs-py`, `alife-defs-cli`

## Language Notes

Research documents are bilingual (Japanese + English). When generating research content, match the language of the target document.
