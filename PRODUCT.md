# PRODUCT.md

## Purpose

ALife Definitions provides an empirical benchmark for comparing competing operational definitions of life. The same digital organisms are evaluated under D1 (7-criteria), D2 (Darwinian/NASA), D3 (autonomy/closure), and D4 (information maintenance) to reveal where and why these definitions agree and disagree.

## Target Users

- ALife researchers studying definitions and criteria of life
- Computational biology and complex systems researchers
- Researchers who want to evaluate their own ALife systems against standardized definition adapters

## Key Features

- Multi-organism continuous 2D simulation substrate
- Config-driven experiment runs with deterministic seeds
- Python analysis layer for definition adapter implementation and scoring
- Criterion-ablation hooks enabling D1 functional-analogy tests
- Experiment summary output for quantitative cross-definition comparison

## Objectives

- Produce publishable benchmark comparing D1–D4 across shared organism traces
- Characterize agreement/disagreement structure across environment regimes
- Provide reusable definition adapters other ALife systems can adopt
- Maintain reproducible simulation runs with held-out test seeds

## Non-Goals (Current Phase)

- Production-grade end-user UI
- Strong ALife claims (the project is framed as weak ALife)
- Stable long-term API guarantees during the research phase
