from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_pairwise import compute_synergy
from scripts.analyze_results import distribution_stats, holm_bonferroni
from scripts.experiment_manifest import config_digest, load_manifest, write_manifest


def test_distribution_stats_non_empty() -> None:
    stats = distribution_stats(np.array([1.0, 2.0, 3.0]))
    assert stats["median"] == 2.0
    assert stats["q25"] <= stats["median"] <= stats["q75"]


def test_holm_bonferroni_preserves_length() -> None:
    corrected = holm_bonferroni([0.01, 0.2, 0.04])
    assert len(corrected) == 3
    assert all(0.0 <= p <= 1.0 for p in corrected)


def test_compute_synergy_sign() -> None:
    assert compute_synergy(10.0, 10.0, 25.0) > 0
    assert compute_synergy(10.0, 10.0, 15.0) < 0


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    base_cfg = {"seed": 42, "mutation_point_rate": 0.02}
    write_manifest(
        path,
        experiment_name="smoke",
        steps=10,
        sample_every=2,
        seeds=[1, 2],
        base_config=base_cfg,
        condition_overrides={"normal": {}},
    )
    loaded = load_manifest(path)
    assert loaded["experiment_name"] == "smoke"
    assert loaded["base_config_digest"] == config_digest(base_cfg)
    assert json.loads(path.read_text())["steps"] == 10
