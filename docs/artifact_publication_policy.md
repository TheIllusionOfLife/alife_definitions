# Artifact Publication Policy

## Overview

This document defines the data archival strategy for the ALife Definitions
Benchmark paper submission to ALIFE 2026.

## Publication Split

| Artifact | Location | License |
|----------|----------|---------|
| Source code (simulation, adapters, analysis) | GitHub | MIT |
| Raw benchmark data (JSON per seed) | Zenodo | CC-BY 4.0 |
| Score matrix TSV + analysis JSONs | GitHub (tracked) | MIT |
| Paper PDF | Venue proceedings | Venue terms |

## Archive Families

Zenodo dataset structure:

```
alife-definitions-benchmark-v1.0/
├── benchmark/
│   ├── E1/seed_000.json ... seed_199.json
│   ├── E2/seed_000.json ... seed_199.json
│   ├── E3/seed_000.json ... seed_199.json
│   ├── E4/seed_000.json ... seed_199.json
│   └── E5/seed_000.json ... seed_199.json
├── score_matrix.tsv
├── agreement_analysis.json
├── predictive_analysis.json
├── frozen_thresholds.json
└── benchmark_manifest.json
```

Estimated size: ~2 GB (5 regimes × 200 seeds × ~2 MB per JSON).

## Paper-Binding References

| Figure/Table | Data Source |
|-------------|-------------|
| Fig 1 (disagreement heatmap) | `score_matrix.tsv` |
| Fig 2 (agreement matrix) | `score_matrix.tsv` → `agreement_analysis.json` |
| Fig 3 (case study) | `benchmark/E1/seed_042.json` |
| Fig 4 (predictive ROC) | `predictive_analysis.json` |

## Submission Sequence Checklist

1. [ ] Run full calibration dataset (seeds 0–99, all regimes)
2. [ ] Run full test dataset (seeds 100–199, all regimes)
3. [ ] Generate score matrix TSV
4. [ ] Run agreement analysis → JSON
5. [ ] Run predictive validity (calibrate on cal, evaluate on test) → JSON
6. [ ] Freeze thresholds → `frozen_thresholds.json`
7. [ ] Generate all 4 figures
8. [ ] Compile paper, verify all figures render
9. [ ] Stage Zenodo archive: `scripts/prepare_zenodo_metadata.py`
10. [ ] Upload to Zenodo (sandbox first): `scripts/upload_zenodo.py`
11. [ ] Publish Zenodo record → get DOI
12. [ ] Update paper with DOI, recompile
13. [ ] Create GitHub Release → triggers code DOI via Zenodo integration
14. [ ] Submit to venue portal
