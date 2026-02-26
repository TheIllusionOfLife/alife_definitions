"""Data contract tests: verify docs/data-contract.md stays in sync with Rust output.

Parses the markdown document to extract documented field names per struct,
runs a minimal simulation, and asserts bidirectional coverage:
  - every documented field exists in the output (no phantom fields)
  - every output field is documented (no undocumented fields)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import alife_defs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data-contract.md"

# Fields that may be absent from raw Rust JSON output:
# - skip_serializing_if = "Vec::is_empty": omitted when the vec is empty
# - Python-layer stamps: added by experiment_common, not by Rust
_CONDITIONALLY_ABSENT = {
    "RunSummary": {"organism_snapshots", "regime_label"},
    "StepMetrics": {"family_breakdown"},
}

_MINIMAL_OVERRIDE = {
    "num_organisms": 4,
    "agents_per_organism": 5,
    "world_size": 20.0,
    "seed": 42,
    "growth_maturation_steps": 10,
    "reproduction_min_energy": 0.31,
    "reproduction_min_boundary": 0.0,
}


# ---------------------------------------------------------------------------
# Markdown field-name parser
# ---------------------------------------------------------------------------


def _parse_documented_fields(section_heading: str) -> set[str]:
    """Extract field names from a table in the data-contract document.

    Looks for a markdown table under *section_heading* (e.g. "### RunSummary")
    and returns the set of field names from the first column.  Field names are
    expected to be in backtick-code spans (e.g. `schema_version`).
    """
    text = _CONTRACT_PATH.read_text()

    # Find the section
    pattern = re.compile(
        rf"^###?\s+{re.escape(section_heading)}\b.*$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    assert match, f"Section '{section_heading}' not found in {_CONTRACT_PATH}"

    # Grab text from the heading to the next heading of same or higher level
    start = match.end()
    next_heading = re.search(r"^(?:#{1,3}\s|---$)", text[start:], re.MULTILINE)
    block = text[start : start + next_heading.start()] if next_heading else text[start:]

    # Extract backtick-wrapped field names from the first column of markdown tables
    # Table rows look like: | `field_name` | type | description |
    fields: set[str] = set()
    for line in block.splitlines():
        m = re.match(r"^\|\s*`(\w+)`\s*\|", line)
        if m:
            fields.add(m.group(1))

    assert fields, f"No fields parsed from section '{section_heading}' — check table format"
    return fields


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _run_mode_a(steps: int = 100, sample_every: int = 10) -> dict:
    """Run a short Mode A simulation and return parsed JSON."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update(_MINIMAL_OVERRIDE)
    return json.loads(alife_defs.run_experiment_json(json.dumps(cfg), steps, sample_every))


def _run_with_snapshots(steps: int = 20, sample_every: int = 10) -> dict:
    """Run a short simulation with snapshot collection and return parsed JSON."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update(_MINIMAL_OVERRIDE)
    snapshot_steps = json.dumps([steps])  # snapshot at last step
    return json.loads(
        alife_defs.run_niche_experiment_json(json.dumps(cfg), steps, sample_every, snapshot_steps)
    )


def _run_mode_b(steps: int = 100, sample_every: int = 10) -> dict:
    """Run a short Mode B (3-family) simulation and return parsed JSON."""
    cfg = json.loads(alife_defs.default_config_json())
    cfg.update({**_MINIMAL_OVERRIDE, "num_organisms": 30})
    cfg["families"] = [
        {
            "enable_metabolism": True,
            "enable_boundary_maintenance": True,
            "enable_homeostasis": True,
            "enable_response": True,
            "enable_reproduction": True,
            "enable_evolution": True,
            "enable_growth": True,
            "initial_count": 10,
            "mutation_rate_multiplier": 1.0,
        },
        {
            "enable_metabolism": True,
            "enable_boundary_maintenance": False,
            "enable_homeostasis": False,
            "enable_response": True,
            "enable_reproduction": True,
            "enable_evolution": True,
            "enable_growth": True,
            "initial_count": 10,
            "mutation_rate_multiplier": 1.0,
        },
        {
            "enable_metabolism": True,
            "enable_boundary_maintenance": True,
            "enable_homeostasis": True,
            "enable_response": True,
            "enable_reproduction": False,
            "enable_evolution": False,
            "enable_growth": True,
            "initial_count": 10,
            "mutation_rate_multiplier": 1.0,
        },
    ]
    return json.loads(alife_defs.run_experiment_json(json.dumps(cfg), steps, sample_every))


# ---------------------------------------------------------------------------
# Contract sync tests
# ---------------------------------------------------------------------------


class TestDocumentExists:
    def test_data_contract_file_exists(self):
        assert _CONTRACT_PATH.exists(), f"Data contract document not found at {_CONTRACT_PATH}"

    def test_schema_version_documented(self):
        text = _CONTRACT_PATH.read_text()
        assert "schema_version" in text


class TestRunSummaryContract:
    """Every RunSummary field must be documented and vice versa."""

    def test_documented_fields_exist_in_output(self):
        doc_fields = _parse_documented_fields("RunSummary")
        skip = _CONDITIONALLY_ABSENT.get("RunSummary", set())
        result = _run_mode_a()
        for field in doc_fields - skip:
            assert field in result, (
                f"Documented RunSummary field `{field}` missing from simulation output"
            )

    def test_output_fields_are_documented(self):
        doc_fields = _parse_documented_fields("RunSummary")
        result = _run_mode_a()
        # regime_label is stamped by Python layer — documented separately
        output_fields = set(result.keys())
        for field in output_fields:
            assert field in doc_fields, (
                f"Undocumented RunSummary field `{field}` found in simulation output"
            )


class TestStepMetricsContract:
    """Every StepMetrics field must be documented and vice versa."""

    def test_documented_fields_exist_in_output(self):
        doc_fields = _parse_documented_fields("StepMetrics")
        skip = _CONDITIONALLY_ABSENT.get("StepMetrics", set())
        result = _run_mode_a()
        assert result["samples"], "Expected at least one sample"
        sample = result["samples"][0]
        for field in doc_fields - skip:
            assert field in sample, (
                f"Documented StepMetrics field `{field}` missing from simulation output"
            )

    def test_output_fields_are_documented(self):
        doc_fields = _parse_documented_fields("StepMetrics")
        result = _run_mode_a()
        sample = result["samples"][0]
        for field in sample:
            assert field in doc_fields, (
                f"Undocumented StepMetrics field `{field}` found in simulation output"
            )


class TestFamilyStepMetricsContract:
    """Every FamilyStepMetrics field must be documented and vice versa."""

    def test_documented_fields_exist_in_output(self):
        doc_fields = _parse_documented_fields("FamilyStepMetrics")
        result = _run_mode_b()
        assert result["samples"], "Expected at least one sample"
        sample = result["samples"][0]
        assert sample.get("family_breakdown"), "Expected family_breakdown in Mode B"
        fam = sample["family_breakdown"][0]
        for field in doc_fields:
            assert field in fam, f"Documented FamilyStepMetrics field `{field}` missing from output"

    def test_output_fields_are_documented(self):
        doc_fields = _parse_documented_fields("FamilyStepMetrics")
        result = _run_mode_b()
        fam = result["samples"][0]["family_breakdown"][0]
        for field in fam:
            assert field in doc_fields, (
                f"Undocumented FamilyStepMetrics field `{field}` found in output"
            )


class TestLineageEventContract:
    """Every LineageEvent field must be documented and vice versa."""

    def test_documented_fields_exist_in_output(self):
        doc_fields = _parse_documented_fields("LineageEvent")
        result = _run_mode_a(steps=500)
        assert result["lineage_events"], "Expected at least one lineage event for contract test"
        event = result["lineage_events"][0]
        for field in doc_fields:
            assert field in event, f"Documented LineageEvent field `{field}` missing from output"

    def test_output_fields_are_documented(self):
        doc_fields = _parse_documented_fields("LineageEvent")
        result = _run_mode_a(steps=500)
        assert result["lineage_events"], "Expected at least one lineage event"
        event = result["lineage_events"][0]
        for field in event:
            assert field in doc_fields, f"Undocumented LineageEvent field `{field}` found in output"


class TestSnapshotContract:
    """OrganismSnapshot and SnapshotFrame fields must be documented."""

    def test_snapshot_frame_fields_bidirectional(self):
        doc_fields = _parse_documented_fields("SnapshotFrame")
        result = _run_with_snapshots()
        assert result["organism_snapshots"], "Expected at least one snapshot frame"
        frame = result["organism_snapshots"][0]
        frame_keys = {k for k in frame if k != "organisms"}  # organisms is a nested list
        frame_keys.add("organisms")  # re-add as documented
        for field in doc_fields:
            assert field in frame, f"Documented SnapshotFrame field `{field}` missing from output"
        for field in frame:
            assert field in doc_fields, (
                f"Undocumented SnapshotFrame field `{field}` found in output"
            )

    def test_organism_snapshot_fields_bidirectional(self):
        doc_fields = _parse_documented_fields("OrganismSnapshot")
        result = _run_with_snapshots()
        assert result["organism_snapshots"], "Expected at least one snapshot frame"
        frame = result["organism_snapshots"][0]
        assert frame["organisms"], "Expected at least one organism in snapshot"
        org = frame["organisms"][0]
        for field in doc_fields:
            assert field in org, f"Documented OrganismSnapshot field `{field}` missing from output"
        for field in org:
            assert field in doc_fields, (
                f"Undocumented OrganismSnapshot field `{field}` found in output"
            )


class TestSimConfigContract:
    """SimConfig fields from default_config_json must all be documented."""

    def test_all_config_fields_documented(self):
        doc_fields = _parse_documented_fields("SimConfig")
        cfg = json.loads(alife_defs.default_config_json())
        for field in cfg:
            assert field in doc_fields, (
                f"Undocumented SimConfig field `{field}` in default_config_json()"
            )

    def test_documented_config_fields_exist(self):
        doc_fields = _parse_documented_fields("SimConfig")
        cfg = json.loads(alife_defs.default_config_json())
        output_fields = set(cfg.keys())
        for field in doc_fields:
            assert field in output_fields, (
                f"Documented SimConfig field `{field}` missing from default_config_json()"
            )


class TestFamilyConfigContract:
    """FamilyConfig fields must be documented."""

    def test_all_family_config_fields_documented(self):
        doc_fields = _parse_documented_fields("FamilyConfig")
        expected = {
            "enable_metabolism",
            "enable_boundary_maintenance",
            "enable_homeostasis",
            "enable_response",
            "enable_reproduction",
            "enable_evolution",
            "enable_growth",
            "initial_count",
            "mutation_rate_multiplier",
        }
        assert doc_fields == expected, (
            f"FamilyConfig doc fields mismatch: {doc_fields.symmetric_difference(expected)}"
        )
