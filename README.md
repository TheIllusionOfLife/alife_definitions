# ALife Definitions

**Benchmarking Multiple Definitions of Life in a Shared Digital World.**

This repository provides a Rust+Python simulation substrate for empirically comparing how competing operational definitions of life agree and disagree when applied to the same digital organisms across different environment regimes.

**Target venue**: ALIFE 2026 Full Paper.

## Quick Start

### Prerequisites

- Rust stable toolchain
- `uv` for Python environment and packaging tasks

### Build

```bash
cargo build -p alife-defs-core -p alife-defs-cli
```

### Test and Lint

```bash
./scripts/check.sh
```

### Python Lint/Test

```bash
uv run ruff check scripts tests_python
uv run pytest tests_python
```

### Build Python Extension (local)

```bash
uv run maturin develop --manifest-path crates/alife-defs-py/Cargo.toml
```

Then in Python:

```python
import alife_defs
print(alife_defs.version())
```

### Run the Benchmark CLI

```bash
cargo run -p alife-defs-cli --release -- benchmark
```

## Repository Docs

- `AGENTS.md`: instructions for coding agents and contributors
- `PRODUCT.md`: product goals and user value
- `TECH.md`: technology stack and technical constraints
- `STRUCTURE.md`: code/documentation layout and conventions
- `docs/README.md`: documentation index
- `docs/research/research-plan.md`: authoritative research plan (new benchmark direction)

## Architecture (High-Level)

- `crates/alife-defs-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/alife-defs-py`: PyO3 bindings exposing core functions to Python
- `crates/alife-defs-cli`: executable benchmark/feasibility runner
- `python/alife_defs`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Current Status

Active research prototype targeting ALIFE 2026. APIs may evolve as the benchmark design is finalized.
