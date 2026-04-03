# ALife Definitions

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19351497.svg)](https://doi.org/10.5281/zenodo.19351497)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19397018.svg)](https://doi.org/10.5281/zenodo.19397018)

**Benchmarking Multiple Definitions of Life in a Shared Digital World.**

This repository provides a Rust+Python simulation substrate for empirically comparing how competing operational definitions of life agree and disagree when applied to the same digital organisms across different environment regimes.

**Paper**: "Life Definitions Disagree: An Empirical Benchmark of Competing Operationalizations in a Shared Digital Ecology" — ALIFE 2026.

## Quick Start

### Prerequisites

- Rust stable toolchain
- `uv` for Python environment and packaging tasks
- `tectonic` for building the paper PDF

### Environment Setup (Locked)

```bash
uv sync --frozen --group dev
```

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
uv run ruff check scripts tests_python python
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

### Reproduce Paper Artifacts

```bash
bash scripts/reproduce_all.sh
```

Notes:
- Fresh rerun is the default behavior; pass `--resume` only when intentionally reusing existing run JSONs.
- Full reproduction is long-running (roughly 30 hours on a Mac Mini M2 Pro with 8 workers).
- Benchmark output is large (around 10 GB); ensure sufficient disk space.
- To refresh predictive artifacts without re-running simulations, run:
  `uv run python scripts/analyze_predictive.py experiments/benchmark -o experiments/benchmark/predictive_analysis.json`
  and
  `uv run python scripts/analyze_predictive.py experiments/benchmark --evaluation-mode strict -o experiments/benchmark/predictive_analysis_strict.json`.

## Architecture (High-Level)

- `crates/alife-defs-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/alife-defs-py`: PyO3 bindings exposing core functions to Python
- `crates/alife-defs-cli`: executable benchmark/feasibility runner
- `python/alife_defs`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{mukai2026life,
  title={Life Definitions Disagree: An Empirical Benchmark of Competing Operationalizations in a Shared Digital Ecology},
  author={Mukai, Yuya},
  booktitle={Artificial Life Conference 2026 (ALIFE 2026)},
  year={2026},
  publisher={MIT Press}
}
```

## License

MIT (code), CC-BY 4.0 (data).
