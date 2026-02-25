# AGENTS.md

This file provides repository-specific instructions for coding agents and contributors.

## Mission

Build and evolve the ALife Definitions benchmark substrate with reproducible, testable changes aligned with the definition-comparison research goals.

## High-Confidence Commands

- Build core + CLI: `cargo build -p alife-defs-core -p alife-defs-cli`
- Build workspace: `cargo build --workspace` (note: alife-defs-py links via maturin, not plain cargo)
- Run full checks: `./scripts/check.sh`
- Run benchmark CLI: `cargo run -p alife-defs-cli --release -- benchmark`
- Build Python extension: `uv run maturin develop --manifest-path crates/alife-defs-py/Cargo.toml`

## Code Style and Quality Rules

- Rust style is default `rustfmt` output with `clippy` warnings treated as errors.
- Keep modules cohesive and domain-driven (`world`, `metabolism`, `organism`, `nn`, `spatial`).
- Add/keep tests close to changed logic (`#[cfg(test)] mod tests`).
- Prefer minimal, direct fixes over broad refactors unless required.

## Testing Instructions

- Baseline gate for all changes: `./scripts/check.sh`
- For Python-binding changes, always run Rust tests in `crates/alife-defs-py/src/lib.rs` through the full test command.

## Repository Etiquette

- Branch naming: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `refactor/<topic>`, `test/<topic>`
- Never push directly to `main`.
- Keep commits small, logically grouped, and prefixed (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`).
- PRs should include the problem, solution, and exact verification commands run.

## Architecture Decisions to Preserve

- Rust core simulation with Python bindings via PyO3/maturin.
- Two metabolism modes are currently supported (`toy`, `graph`).
- Criterion-ablation toggles are config-driven and should stay easy to test.
- Simulation behavior should remain reproducible via deterministic seeds.

## Environment and Tooling Quirks

- Use `uv` for Python-related tooling; avoid ad-hoc global package installs.
- Running `cargo run -p alife-defs-cli` without `--release` yields non-representative performance.
- Local build artifacts (`target/`, extension binaries) should remain untracked.

## Non-Obvious Gotchas

- `num_organisms * agents_per_organism` is bounded; exceeding limits is rejected by runtime checks.
- JSON config compatibility matters: legacy config payloads should still deserialize with defaults when possible.
- Research planning docs are in `docs/research/`; do not treat them as implementation truth over code/tests.
