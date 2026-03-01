#!/bin/bash
# Run all benchmark data generation for the paper revision.
# Usage: nohup ./scripts/run_all_benchmarks.sh > experiments/run_all.log 2>&1 &
#
# Both jobs use --resume so they skip already-completed seeds.
# Safe to restart at any time.

set -euo pipefail
cd "$(dirname "$0")/.."

# Shared parameters — generates raw data for both calibration (0-99) and
# held-out test (100-199) seeds.  Threshold calibration vs evaluation split
# is enforced downstream by analyze_predictive.py, not here.
SEEDS="0-199"
REGIMES="E1,E2,E3,E4,E5"

echo "=== Mode B benchmark (1000 runs) ==="
uv run python scripts/experiment_benchmark.py \
    --seeds "$SEEDS" --regimes "$REGIMES" --resume

echo ""
echo "=== Single-family controls (3000 runs) ==="
uv run python scripts/experiment_benchmark_single.py \
    --seeds "$SEEDS" --regimes "$REGIMES" --resume

echo ""
echo "=== All benchmarks complete ==="
