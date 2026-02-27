#!/usr/bin/env bash
# reproduce_all.sh — Run the full benchmark pipeline from scratch.
#
# Usage:
#   bash scripts/reproduce_all.sh [--seeds 0-4] [--regimes E1,E2]
#
# Default: calibration seeds 0-99 + test seeds 100-199, all 5 regimes.
# For quick testing, use: bash scripts/reproduce_all.sh --seeds 0-4 --regimes E1
set -euo pipefail

SEEDS="${SEEDS:-0-99}"
TEST_SEEDS="${TEST_SEEDS:-100-199}"
REGIMES="${REGIMES:-E1,E2,E3,E4,E5}"
DATA_DIR="experiments/benchmark"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --seeds) SEEDS="$2"; shift 2 ;;
    --test-seeds) TEST_SEEDS="$2"; shift 2 ;;
    --regimes) REGIMES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "=== ALife Definitions Benchmark Pipeline ==="
echo "  Calibration seeds: ${SEEDS}"
echo "  Test seeds: ${TEST_SEEDS}"
echo "  Regimes: ${REGIMES}"
echo ""

# Step 1: Generate benchmark data (calibration)
echo "--- Step 1: Generate calibration data ---"
uv run python -m scripts.experiment_benchmark \
  --seeds "${SEEDS}" --regimes "${REGIMES}" --resume

# Step 2: Generate benchmark data (test)
echo "--- Step 2: Generate test data ---"
uv run python -m scripts.experiment_benchmark \
  --seeds "${TEST_SEEDS}" --regimes "${REGIMES}" --resume

# Step 3: Score all runs
echo "--- Step 3: Score matrix ---"
uv run python -m scripts.score_benchmark \
  --data-dir "${DATA_DIR}" \
  --seeds "${SEEDS},${TEST_SEEDS}" \
  --regimes "${REGIMES}" \
  > "${DATA_DIR}/score_matrix.tsv"
echo "  Wrote ${DATA_DIR}/score_matrix.tsv"

# Step 4: Agreement analysis
echo "--- Step 4: Agreement analysis ---"
uv run python scripts/analyze_agreement.py \
  "${DATA_DIR}/score_matrix.tsv" \
  -o "${DATA_DIR}/agreement_analysis.json"
echo "  Wrote ${DATA_DIR}/agreement_analysis.json"

# Step 5: Predictive validity
echo "--- Step 5: Predictive validity ---"
uv run python scripts/analyze_predictive.py \
  "${DATA_DIR}" \
  --cal-seeds "${SEEDS}" \
  --test-seeds "${TEST_SEEDS}" \
  --regimes "${REGIMES}" \
  -o "${DATA_DIR}/predictive_analysis.json"
echo "  Wrote ${DATA_DIR}/predictive_analysis.json"

# Step 6: Generate figures
echo "--- Step 6: Figures ---"
CASE_JSON="${DATA_DIR}/E1/seed_042.json"
if [ ! -f "${CASE_JSON}" ]; then
  CASE_JSON=""
fi
uv run python scripts/figure_agreement.py \
  "${DATA_DIR}/score_matrix.tsv" \
  ${CASE_JSON:+--case-study-json "${CASE_JSON}"} \
  --predictive-json "${DATA_DIR}/predictive_analysis.json"

# Step 7: Compile paper
echo "--- Step 7: Compile paper ---"
cd paper && tectonic main.tex && cd ..
echo "  Paper: paper/main.pdf"

echo ""
echo "=== Pipeline complete ==="
