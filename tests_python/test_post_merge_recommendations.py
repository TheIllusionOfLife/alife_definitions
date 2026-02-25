from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import analyses.coupling.transfer_entropy as _te_module
import numpy as np
import pytest

import scripts.analyze_coupling as analyze_coupling
from scripts.analyze_coupling import phase_randomize, te_robustness_summary
from scripts.experiment_manifest import load_manifest, write_manifest


def test_manifest_schema_v2_supports_report_bindings(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        experiment_name="final_graph_ablation",
        steps=2000,
        sample_every=50,
        seeds=[100, 101],
        base_config={"seed": 100, "mutation_point_rate": 0.02},
        condition_overrides={"normal": {}},
        report_bindings=[
            {
                "result_id": "coupling_main",
                "paper_ref": "fig:coupling",
                "source_files": [
                    "experiments/final_graph_normal.json",
                    "experiments/coupling_analysis.json",
                ],
                "notes": "Primary coupling claim source",
            }
        ],
    )

    payload = load_manifest(path)
    assert payload["schema_version"] == 2
    assert payload["report_bindings"][0]["paper_ref"] == "fig:coupling"


def test_persistence_claim_gate_threshold() -> None:
    # Imported inline to keep this module decoupled from analyze_phenotype load-time deps.
    from scripts.analyze_phenotype import persistence_claim_gate

    assert persistence_claim_gate(0.2999, threshold=0.30) is False
    assert persistence_claim_gate(0.3000, threshold=0.30) is True


def test_te_robustness_summary_shape() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    y = 0.2 * np.roll(x, 1) + rng.normal(size=120)
    rows = te_robustness_summary(
        x,
        y,
        bin_settings=[3],
        permutation_settings=[20],
        rng_seed=7,
        phase_surrogate_samples=8,
        surrogate_permutation_floor=8,
        surrogate_permutation_divisor=2,
    )

    assert len(rows) == 1
    assert rows[0]["bins"] == 3
    assert rows[0]["permutations"] == 20
    assert "te" in rows[0]
    assert "p_value" in rows[0]


def test_te_robustness_summary_excludes_none_surrogates(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_te(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"te": 0.4, "p_value": 0.2, "null_mean": 0.0, "null_std": 0.0}
        if call_count["n"] in (3, 5):
            return None
        return {"te": 0.1, "p_value": 0.3, "null_mean": 0.0, "null_std": 0.0}

    monkeypatch.setattr(_te_module, "transfer_entropy_lag1", fake_te)

    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    rows = te_robustness_summary(
        x,
        y,
        bin_settings=[3],
        permutation_settings=[20],
        rng_seed=7,
        phase_surrogate_samples=4,
        surrogate_permutation_floor=8,
        surrogate_permutation_divisor=2,
    )
    assert len(rows) == 1
    assert rows[0]["phase_surrogate_valid_n"] == 2
    assert rows[0]["phase_surrogate_te_mean"] == 0.1


def test_phase_randomize_even_keeps_nyquist_bin_real_and_fixed() -> None:
    rng = np.random.default_rng(123)
    series = np.random.default_rng(9).normal(size=8)
    before = np.fft.rfft(series)
    after = np.fft.rfft(phase_randomize(series, rng))

    assert np.isclose(after[0].imag, 0.0, atol=1e-10)
    assert np.isclose(after[-1], before[-1], atol=1e-8)


def test_phase_randomize_odd_randomizes_last_complex_bin_phase() -> None:
    rng = np.random.default_rng(456)
    series = np.random.default_rng(10).normal(size=9)
    before = np.fft.rfft(series)
    after = np.fft.rfft(phase_randomize(series, rng))

    assert np.isclose(abs(after[-1]), abs(before[-1]), atol=1e-8)
    assert not np.isclose(after[-1], before[-1], atol=1e-8)


def test_phase_randomize_small_n_returns_copy() -> None:
    rng = np.random.default_rng(777)
    series = np.array([0.1, -0.2, 0.3], dtype=float)
    out = phase_randomize(series, rng)

    assert out.dtype == float
    assert len(out) == len(series)
    assert np.allclose(out, series)
    assert out is not series


def test_main_rejects_unknown_robustness_profile() -> None:
    with pytest.raises(ValueError):
        analyze_coupling.main(robustness_profile="unknown")


def test_experiment_niche_defaults_and_long_horizon_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = types.SimpleNamespace()
    calls: list[tuple[int, int, list[int]]] = []

    def fake_version() -> str:
        return "test"

    def fake_run_niche(
        config_json: str, steps: int, sample_every: int, snapshot_steps_json: str
    ) -> str:
        _ = config_json
        calls.append((steps, sample_every, json.loads(snapshot_steps_json)))
        payload = {"final_alive_count": 1, "organism_snapshots": [{"organisms": [{"id": 1}]}]}
        return json.dumps(payload)

    fake.version = fake_version
    fake.run_niche_experiment_json = fake_run_niche
    monkeypatch.setitem(sys.modules, "alife_defs", fake)
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))

    mod = importlib.import_module("scripts.experiment_niche")
    mod = importlib.reload(mod)

    assert mod.SEEDS == list(range(100, 130))

    monkeypatch.setattr(mod, "SEEDS", [100])
    monkeypatch.setattr(mod, "make_config", lambda seed, overrides: "{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_niche.py", "--long-horizon", "--output", str(tmp_path / "custom_long.json")],
    )
    mod.main()
    assert calls[-1][0] == mod.LONG_HORIZON_STEPS
    assert calls[-1][1] == mod.SAMPLE_EVERY
    assert calls[-1][2] == mod.LONG_HORIZON_SNAPSHOT_STEPS
    assert (tmp_path / "custom_long.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_niche.py", "--output", str(tmp_path / "niche_normal.json")],
    )
    mod.main()
    assert calls[-1][0] == mod.STEPS
    assert calls[-1][1] == mod.SAMPLE_EVERY
    assert calls[-1][2] == mod.SNAPSHOT_STEPS
    assert (tmp_path / "niche_normal.json").exists()


def test_experiment_regimes_seed_count_is_n30(monkeypatch: pytest.MonkeyPatch) -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "experiment_regimes.py"

    fake_alife_defs = types.SimpleNamespace(version=lambda: "test")
    monkeypatch.setitem(sys.modules, "alife_defs", fake_alife_defs)
    monkeypatch.syspath_prepend(str(script_dir))

    spec = importlib.util.spec_from_file_location("experiment_regimes_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "SEEDS")
    assert len(module.SEEDS) == 30
    assert module.SEEDS[0] == 100
    assert module.SEEDS[-1] == 129


def test_experiment_niche_seed_range_batching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = types.SimpleNamespace()
    seen_seeds: list[int] = []

    def fake_version() -> str:
        return "test"

    def fake_run_niche(
        config_json: str, steps: int, sample_every: int, snapshot_steps_json: str
    ) -> str:
        _ = steps, sample_every, snapshot_steps_json
        seen_seeds.append(json.loads(config_json)["seed"])
        return json.dumps({"final_alive_count": 1, "organism_snapshots": []})

    fake.version = fake_version
    fake.run_niche_experiment_json = fake_run_niche
    monkeypatch.setitem(sys.modules, "alife_defs", fake)
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))

    mod = importlib.import_module("scripts.experiment_niche")
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "make_config", lambda seed, overrides: json.dumps({"seed": seed}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment_niche.py",
            "--seed-start",
            "105",
            "--seed-end",
            "107",
            "--output",
            str(tmp_path / "batch.json"),
        ],
    )
    mod.main()

    assert seen_seeds == [105, 106, 107]
    assert (tmp_path / "batch.json").exists()


def test_experiment_niche_rejects_invalid_seed_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace()
    fake.version = lambda: "test"
    fake.run_niche_experiment_json = lambda *_args, **_kwargs: json.dumps(
        {"final_alive_count": 1, "organism_snapshots": []}
    )
    monkeypatch.setitem(sys.modules, "alife_defs", fake)
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))

    mod = importlib.import_module("scripts.experiment_niche")
    mod = importlib.reload(mod)

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_niche.py", "--seed-start", "110", "--seed-end", "109"],
    )
    with pytest.raises(SystemExit):
        mod.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_niche.py", "--seed-start", "99", "--seed-end", "109"],
    )
    with pytest.raises(SystemExit):
        mod.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_niche.py", "--seed-start", "190", "--seed-end", "200"],
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_analyze_phenotype_long_horizon_sensitivity(tmp_path: Path) -> None:
    from scripts.analyze_phenotype import analyze_long_horizon_sensitivity

    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True)

    def make_seed(seed: int, steps: list[int], stable_ids: list[int]) -> dict:
        frames = []
        for step in steps:
            frames.append(
                {
                    "step": step,
                    "organisms": [
                        {
                            "stable_id": sid,
                            "energy": 1.0 + sid * 0.01,
                            "waste": 0.2,
                            "boundary_integrity": 0.8,
                            "maturity": 1.0,
                            "generation": sid % 3,
                        }
                        for sid in stable_ids
                    ],
                }
            )
        return {"seed": seed, "organism_snapshots": frames}

    standard = [make_seed(100, [2000, 2200, 4500, 4700], [1, 2, 3, 4, 5])]
    long_horizon = [
        make_seed(100, [2000, 2200, 4500, 4700, 7000, 7200, 9500, 9700], [1, 2, 3, 4, 5])
    ]

    (exp_dir / "niche_normal.json").write_text(json.dumps(standard))
    (exp_dir / "niche_normal_long.json").write_text(json.dumps(long_horizon))

    out = analyze_long_horizon_sensitivity(exp_dir)
    assert out["available"] is True
    assert out["long_horizon_path"].endswith("niche_normal_long.json")
    assert "adjusted_rand_index" in out["comparison"]
