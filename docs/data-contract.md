# Data Contract — alife-defs Simulation Output Schema

> **Schema version**: 1
> **Status**: Frozen. This document is the authoritative reference for all adapter and analysis code.
> **Validated by**: `tests_python/test_data_contract.py` (bidirectional sync test)

## Stability Policy

- **Adding** new optional fields with `#[serde(default)]` is allowed without bumping the version.
- **Removing or renaming** existing fields requires incrementing `schema_version`.
- The Python test suite enforces that this document and the Rust output stay in sync.

---

## Output Structs

### RunSummary

Top-level object returned by `run_experiment_json()`.

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `u32` | Schema version number (currently `1`) |
| `steps` | `usize` | Total simulation steps executed |
| `sample_every` | `usize` | Sampling interval (metrics collected every N steps) |
| `final_alive_count` | `usize` | Number of alive organisms at simulation end |
| `samples` | `Vec<StepMetrics>` | Time-series of per-step population metrics |
| `lifespans` | `Vec<usize>` | Lifespan (in steps) of each organism that died during the run |
| `total_reproduction_events` | `usize` | Cumulative reproduction events across the run |
| `lineage_events` | `Vec<LineageEvent>` | Parent-child lineage records |
| `organism_snapshots` | `Vec<SnapshotFrame>` | Optional per-organism spatial snapshots (omitted when empty) |

**Python-layer additions** (stamped by `experiment_common.run_single()`):

| Field | Type | Description |
|-------|------|-------------|
| `regime_label` | `str` | Environment regime identifier (e.g. `"E1_baseline"`, `""` if unset) |

### StepMetrics

Per-step population aggregate. One entry per `sample_every` steps.

| Field | Type | Description |
|-------|------|-------------|
| `step` | `usize` | Simulation step number |
| `energy_mean` | `f32` | Mean energy across alive organisms |
| `energy_std` | `f32` | Standard deviation of energy |
| `waste_mean` | `f32` | Mean waste level |
| `waste_std` | `f32` | Standard deviation of waste |
| `boundary_mean` | `f32` | Mean boundary integrity |
| `boundary_std` | `f32` | Standard deviation of boundary integrity |
| `alive_count` | `usize` | Number of currently alive organisms |
| `population_size` | `usize` | Total organism slots (alive + dead/recycled) |
| `birth_count` | `usize` | Births since last sample |
| `death_count` | `usize` | Deaths since last sample |
| `resource_total` | `f64` | Total resource in the environment |
| `mean_age` | `f32` | Mean age (in steps) of alive organisms |
| `mean_generation` | `f32` | Mean generation number |
| `max_generation` | `usize` | Highest generation observed among alive organisms |
| `mean_genome_drift` | `f32` | Mean L1 drift from ancestor genome |
| `genome_diversity` | `f32` | Mean pairwise L2 distance between alive organism genomes |
| `internal_state_mean` | `[f32; 4]` | Mean internal state across alive agents (4 channels) |
| `internal_state_std` | `[f32; 4]` | Standard deviation of internal state (4 channels) |
| `maturity_mean` | `f32` | Mean developmental maturity (0=newborn, 1=mature) |
| `spatial_cohesion_mean` | `f32` | Mean pairwise agent distance per organism (lower = tighter) |
| `agent_id_exhaustion_events` | `usize` | Agent ID pool exhaustion events this step |
| `family_breakdown` | `Vec<FamilyStepMetrics>` | Per-family metrics; empty in Mode A, one per family in Mode B |

### FamilyStepMetrics

Per-family population metrics within a single sample step (Mode B only).

**Invariant**: `sum(family.alive_count) == global.alive_count` for each step.

| Field | Type | Description |
|-------|------|-------------|
| `family_id` | `u16` | Family index (0-based) |
| `alive_count` | `usize` | Alive organisms in this family |
| `population_size` | `usize` | Total organism slots for this family |
| `energy_mean` | `f32` | Mean energy |
| `waste_mean` | `f32` | Mean waste level |
| `boundary_mean` | `f32` | Mean boundary integrity |
| `birth_count` | `usize` | Births this step |
| `death_count` | `usize` | Deaths this step |
| `mean_generation` | `f32` | Mean generation number |
| `mean_genome_drift` | `f32` | Mean L1 drift from ancestor genome |
| `genome_diversity` | `f32` | Mean pairwise L2 genome distance |
| `maturity_mean` | `f32` | Mean developmental maturity |

### LineageEvent

Recorded for each reproduction event.

| Field | Type | Description |
|-------|------|-------------|
| `step` | `usize` | Simulation step of reproduction |
| `parent_stable_id` | `u64` | Stable ID of parent organism |
| `child_stable_id` | `u64` | Stable ID of child organism |
| `generation` | `u32` | Child's generation number |
| `genome_hash` | `u64` | FNV-1a hash of child genome vector bytes |
| `family_id` | `u16` | Child's family index (0 in Mode A) |
| `parent_genome_hash` | `u64` | FNV-1a hash of parent genome vector bytes |
| `parent_child_genome_distance` | `f32` | Normalized L2 distance between parent and child genomes: `sqrt(mean((p_i - c_i)²))` |

### SnapshotFrame

Optional spatial snapshot at a specific step (only present when `snapshot_steps` is requested via `run_niche_experiment_json`).

| Field | Type | Description |
|-------|------|-------------|
| `step` | `usize` | Simulation step |
| `organisms` | `Vec<OrganismSnapshot>` | Per-organism spatial data |

### OrganismSnapshot

Per-organism state within a snapshot frame.

| Field | Type | Description |
|-------|------|-------------|
| `stable_id` | `u64` | Stable organism identifier |
| `generation` | `u32` | Generation number |
| `age_steps` | `usize` | Age in simulation steps |
| `energy` | `f32` | Current energy level |
| `waste` | `f32` | Current waste level |
| `boundary_integrity` | `f32` | Current boundary integrity |
| `maturity` | `f32` | Developmental maturity (0–1) |
| `center_x` | `f64` | X coordinate of organism center |
| `center_y` | `f64` | Y coordinate of organism center |
| `n_agents` | `usize` | Number of agents belonging to this organism |

---

## Input Schema

### SimConfig

Full simulation configuration. Obtained via `default_config_json()`, validated by `validate_config_json()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `u64` | `42` | Deterministic RNG seed |
| `world_size` | `f64` | `100.0` | Width/height of square toroidal world |
| `num_organisms` | `usize` | `50` | Number of organisms |
| `agents_per_organism` | `usize` | `50` | Agents per organism |
| `sensing_radius` | `f64` | `5.0` | Local neighbor sensing radius |
| `max_speed` | `f64` | `2.0` | Agent velocity clamp |
| `dt` | `f64` | `0.1` | Simulation timestep |
| `neighbor_norm` | `f64` | `50.0` | Neighbor-count NN input normalization |
| `enable_metabolism` | `bool` | `true` | Criterion ablation: metabolism |
| `enable_boundary_maintenance` | `bool` | `true` | Criterion ablation: boundary maintenance |
| `enable_homeostasis` | `bool` | `true` | Criterion ablation: homeostasis |
| `enable_response` | `bool` | `true` | Criterion ablation: stimulus response |
| `enable_reproduction` | `bool` | `true` | Criterion ablation: reproduction |
| `enable_evolution` | `bool` | `true` | Criterion ablation: evolution |
| `enable_growth` | `bool` | `true` | Criterion ablation: growth/development |
| `ablation_step` | `usize` | `0` | Step to apply scheduled ablation (0=disabled) |
| `ablation_targets` | `Vec<AblationTarget>` | `[]` | Criteria to ablate at `ablation_step` |
| `boundary_mode` | `BoundaryMode` | `"scalar_repair"` | Boundary implementation mode |
| `homeostasis_mode` | `HomeostasisMode` | `"nn_regulator"` | Homeostasis implementation mode |
| `setpoint_pid_base` | `f32` | `0.45` | PID base setpoint |
| `setpoint_pid_energy_scale` | `f32` | `0.1` | PID energy scaling |
| `setpoint_pid_kp` | `f32` | `0.5` | PID proportional gain |
| `spatial_hull_repair_base` | `f32` | `0.6` | SpatialHullFeedback repair base |
| `spatial_hull_repair_cohesion_scale` | `f32` | `0.8` | Cohesion multiplier for repair |
| `spatial_hull_decay_base` | `f32` | `1.2` | SpatialHullFeedback decay base |
| `spatial_hull_decay_cohesion_scale` | `f32` | `0.5` | Cohesion multiplier subtracted from decay |
| `spatial_hull_decay_min` | `f32` | `0.5` | Minimum decay scaling |
| `metabolic_viability_floor` | `f32` | `0.2` | Minimum energy for stable boundary maintenance |
| `boundary_decay_base_rate` | `f32` | `0.001` | Baseline boundary decay per step |
| `boundary_decay_energy_scale` | `f32` | `0.02` | Additional decay from low energy |
| `boundary_waste_pressure_scale` | `f32` | `0.5` | Waste weight in boundary pressure |
| `boundary_repair_waste_penalty_scale` | `f32` | `0.4` | Waste penalty in repair effectiveness |
| `boundary_repair_rate` | `f32` | `0.05` | Per-step boundary repair multiplier |
| `boundary_collapse_threshold` | `f32` | `0.05` | Boundary below which organism collapses |
| `death_energy_threshold` | `f32` | `0.0` | Energy threshold for terminal viability |
| `death_boundary_threshold` | `f32` | `0.1` | Boundary threshold for terminal viability |
| `metabolism_mode` | `MetabolismMode` | `"toy"` | Metabolism engine mode |
| `reproduction_min_energy` | `f32` | `0.85` | Minimum energy to reproduce |
| `reproduction_min_boundary` | `f32` | `0.70` | Minimum boundary to reproduce |
| `reproduction_energy_cost` | `f32` | `0.30` | Energy deducted during reproduction |
| `reproduction_child_min_agents` | `usize` | `4` | Minimum agents for child organism |
| `reproduction_spawn_radius` | `f64` | `1.0` | Child agent spawn radius |
| `crowding_neighbor_threshold` | `f32` | `8.0` | Neighbor density for crowding damage |
| `crowding_boundary_decay` | `f32` | `0.0015` | Boundary decay from crowding |
| `max_organism_age_steps` | `usize` | `20000` | Forced death age limit |
| `compaction_interval_steps` | `usize` | `64` | Dead entity pruning interval |
| `mutation_point_rate` | `f32` | `0.02` | Per-gene point mutation probability |
| `mutation_point_scale` | `f32` | `0.15` | Point mutation delta bound |
| `mutation_reset_rate` | `f32` | `0.002` | Per-gene reset-to-zero probability |
| `mutation_scale_rate` | `f32` | `0.002` | Per-gene multiplicative scale probability |
| `mutation_scale_min` | `f32` | `0.8` | Minimum scale mutation factor |
| `mutation_scale_max` | `f32` | `1.2` | Maximum scale mutation factor |
| `mutation_value_limit` | `f32` | `2.0` | Absolute clamp for genome values |
| `homeostasis_decay_rate` | `f32` | `0.01` | Per-step internal state decay |
| `growth_maturation_steps` | `usize` | `200` | Steps for child to reach maturity |
| `growth_immature_metabolic_efficiency` | `f32` | `0.3` | Metabolic efficiency at maturity=0 |
| `resource_regeneration_rate` | `f32` | `0.01` | Per-cell resource regeneration rate |
| `environment_shift_step` | `usize` | `0` | Step for environment shift (0=disabled) |
| `environment_shift_resource_rate` | `f32` | `0.01` | Post-shift resource regeneration rate |
| `metabolism_efficiency_multiplier` | `f32` | `1.0` | Graded metabolic ablation (0–1) |
| `environment_cycle_period` | `usize` | `0` | Cyclic resource modulation period (0=off) |
| `environment_cycle_low_rate` | `f32` | `0.005` | Low-phase resource regeneration rate |
| `enable_sham_process` | `bool` | `false` | Sham (no-op) computational process toggle |
| `sensing_noise_scale` | `f32` | `0.0` | E4: Gaussian noise stddev on NN sensory inputs |
| `resource_patch_count` | `usize` | `0` | E5: number of high-resource patches (0=uniform) |
| `resource_patch_scale` | `f32` | `1.0` | E5: peak regeneration multiplier at patch centres |
| `families` | `Vec<FamilyConfig>` | `[]` | Mode B family configurations (empty=Mode A) |

**Validation bounds**: `MAX_WORLD_SIZE = 2048.0`, `MAX_TOTAL_AGENTS = 250,000`, `MAX_RESOURCE_PATCH_COUNT = 256`.

### FamilyConfig

Per-family criterion ablation and population parameters (Mode B).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_metabolism` | `bool` | `true` | Metabolism toggle |
| `enable_boundary_maintenance` | `bool` | `true` | Boundary maintenance toggle |
| `enable_homeostasis` | `bool` | `true` | Homeostasis toggle |
| `enable_response` | `bool` | `true` | Stimulus response toggle |
| `enable_reproduction` | `bool` | `true` | Reproduction toggle |
| `enable_evolution` | `bool` | `true` | Evolution toggle |
| `enable_growth` | `bool` | `true` | Growth/development toggle |
| `initial_count` | `usize` | `10` | Organisms seeded for this family |
| `mutation_rate_multiplier` | `f32` | `1.0` | Multiplied against global mutation rate |

### Enum Types

**MetabolismMode**: `"toy"` | `"graph"` | `"counter"`

**BoundaryMode**: `"scalar_repair"` | `"spatial_hull_feedback"`

**HomeostasisMode**: `"nn_regulator"` | `"setpoint_pid"`

**AblationTarget**: `"metabolism"` | `"boundary"` | `"homeostasis"` | `"response"` | `"reproduction"` | `"evolution"` | `"growth"`

---

## Environment Regimes

| Regime | Label | Key Overrides |
|--------|-------|---------------|
| E1 | Baseline | No overrides (default config) |
| E2 | Sparse resources | `resource_regeneration_rate: 0.005`, `world_size: 150.0` |
| E3 | Crowded | `num_organisms: 80`, `agents_per_organism: 30`, `world_size: 80.0` |
| E4 | Sensing noise | `sensing_noise_scale: 0.5` |
| E5 | Spatial patchiness | `resource_patch_count: 4`, `resource_patch_scale: 2.0` |

---

## Mode A vs Mode B

- **Mode A** (single-family): `families` is `[]`. All organisms share global `enable_*` flags. `family_breakdown` in StepMetrics is empty (omitted from JSON via `skip_serializing_if`).
- **Mode B** (multi-family): `families` has 1+ entries. Each family has its own `enable_*` flags. `family_breakdown` contains one `FamilyStepMetrics` per family at each sample step. `LineageEvent.family_id` distinguishes lineage across families.

---

## Annotated Example

Minimal 10-step Mode B run with 2 families, sampled every 5 steps (2 sample points, truncated for readability):

```json
{
  "schema_version": 1,
  "steps": 10,
  "sample_every": 5,
  "final_alive_count": 18,
  "samples": [
    {
      "step": 5,
      "energy_mean": 0.235,
      "energy_std": 0.130,
      "waste_mean": 0.010,
      "waste_std": 0.005,
      "boundary_mean": 0.997,
      "boundary_std": 0.002,
      "alive_count": 18,
      "population_size": 18,
      "birth_count": 0,
      "death_count": 0,
      "resource_total": 397.63,
      "mean_age": 3.33,
      "mean_generation": 0.667,
      "max_generation": 1,
      "mean_genome_drift": 0.001,
      "genome_diversity": 10.46,
      "internal_state_mean": [0.407, 0.409, 0.999, 0.500],
      "internal_state_std": [0.160, 0.140, 0.001, 0.000],
      "maturity_mean": 0.500,
      "spatial_cohesion_mean": 2.426,
      "agent_id_exhaustion_events": 0,
      "family_breakdown": [
        {
          "family_id": 0,
          "alive_count": 9,
          "population_size": 9,
          "energy_mean": 0.235,
          "waste_mean": 0.010,
          "boundary_mean": 0.997,
          "birth_count": 0,
          "death_count": 0,
          "mean_generation": 0.667,
          "mean_genome_drift": 0.001,
          "genome_diversity": 8.738,
          "maturity_mean": 0.500
        },
        {
          "family_id": 1,
          "alive_count": 9,
          "population_size": 9,
          "energy_mean": 0.235,
          "waste_mean": 0.010,
          "boundary_mean": 0.998,
          "birth_count": 0,
          "death_count": 0,
          "mean_generation": 0.667,
          "mean_genome_drift": 0.001,
          "genome_diversity": 9.083,
          "maturity_mean": 0.500
        }
      ]
    }
  ],
  "lifespans": [],
  "total_reproduction_events": 0,
  "lineage_events": [],
  "organism_snapshots": []
}
```
