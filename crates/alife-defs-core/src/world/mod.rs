use crate::agent::Agent;
use crate::config::{AblationTarget, FamilyConfig, MetabolismMode, SimConfig, SimConfigError};
use crate::genome::{Genome, MutationRates};
use crate::metabolism::{MetabolicState, MetabolismEngine};
use crate::nn::NeuralNet;
use crate::organism::{DevelopmentalProgram, OrganismRuntime};
use crate::resource::ResourceField;
use crate::spatial;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::time::Instant;
use std::{error::Error, fmt};

use crate::metrics::{LineageEvent, OrganismSnapshot, PopulationStats, RunSummary, SnapshotFrame};

/// Decode a genome's metabolic segment into a per-organism `MetabolismEngine`.
///
/// Returns `Some(engine)` in Graph mode, `None` in Toy/Counter mode (uses shared engine).
fn decode_organism_metabolism(genome: &Genome, mode: MetabolismMode) -> Option<MetabolismEngine> {
    match mode {
        MetabolismMode::Graph => {
            let gm = crate::metabolism::decode_graph_metabolism(genome.segment_data(1));
            Some(MetabolismEngine::Graph(gm))
        }
        MetabolismMode::Toy | MetabolismMode::Counter => None,
    }
}

#[derive(Clone, Debug)]
pub struct StepTimings {
    pub spatial_build_us: u64,
    pub nn_query_us: u64,
    pub state_update_us: u64,
    pub total_us: u64,
}

pub struct World {
    agents: Vec<Agent>,
    organisms: Vec<OrganismRuntime>,
    config: SimConfig,
    metabolism: MetabolismEngine,
    resource_field: ResourceField,
    org_toroidal_sums: Vec<[f64; 4]>,
    org_counts: Vec<usize>,
    rng: ChaCha12Rng,
    next_agent_id: u32,
    step_index: usize,
    original_config: Option<SimConfig>,
    scheduled_ablation_applied: bool,
    births_last_step: usize,
    deaths_last_step: usize,
    births_per_family: Vec<usize>,
    deaths_per_family: Vec<usize>,
    total_births: usize,
    total_deaths: usize,
    mutation_rates: MutationRates,
    next_organism_stable_id: u64,
    agent_id_exhaustions_last_step: usize,
    total_agent_id_exhaustions: usize,
    lifespans: Vec<usize>,
    lineage_events: Vec<LineageEvent>,
    /// Runtime resource regeneration rate, separate from config to avoid mutating
    /// config at runtime during environment shifts.
    current_resource_rate: f32,

    // Buffers for avoiding allocation in simulation steps
    deltas_buffer: Vec<[f32; 4]>,
    neighbor_sums_buffer: Vec<f32>,
    neighbor_counts_buffer: Vec<usize>,
    homeostasis_sums_buffer: Vec<f32>,
    homeostasis_counts_buffer: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorldInitError {
    Config(SimConfigError),
    AgentCountOverflow,
    TooManyAgents {
        max: usize,
        actual: usize,
    },
    NumOrganismsMismatch {
        expected: usize,
        actual: usize,
    },
    AgentCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidOrganismId,
    /// families[*].initial_count sum does not match num_organisms.
    FamilyOrganismCountMismatch {
        expected: usize,
        actual: usize,
    },
    /// families.len() exceeds u16::MAX; family_id is stored as u16.
    TooManyFamilies {
        actual: usize,
    },
}

impl fmt::Display for WorldInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorldInitError::Config(e) => write!(f, "{}", e),
            WorldInitError::AgentCountOverflow => {
                write!(f, "num_organisms * agents_per_organism overflows usize")
            }
            WorldInitError::TooManyAgents { max, actual } => {
                write!(f, "total agents ({actual}) exceeds supported maximum ({max})")
            }
            WorldInitError::NumOrganismsMismatch { expected, actual } => write!(
                f,
                "num_organisms ({expected}) must match nns.len() ({actual})"
            ),
            WorldInitError::AgentCountMismatch { expected, actual } => write!(
                f,
                "agents.len() ({actual}) must match num_organisms * agents_per_organism ({expected})"
            ),
            WorldInitError::InvalidOrganismId => {
                write!(f, "all agent organism_ids must be valid indices into nns")
            }
            WorldInitError::FamilyOrganismCountMismatch { expected, actual } => write!(
                f,
                "sum of families[*].initial_count ({expected}) must equal num_organisms \
                 ({actual}); fix by setting num_organisms = {expected} or adjusting \
                 FamilyConfig.initial_count values"
            ),
            WorldInitError::TooManyFamilies { actual } => write!(
                f,
                "families.len() ({actual}) exceeds u16::MAX ({}); \
                 family_id is stored as u16",
                u16::MAX
            ),
        }
    }
}

impl From<SimConfigError> for WorldInitError {
    fn from(err: SimConfigError) -> Self {
        WorldInitError::Config(err)
    }
}

impl Error for WorldInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WorldInitError::Config(e) => Some(e),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    InvalidSampleEvery,
    TooManySteps { max: usize, actual: usize },
    TooManySamples { max: usize, actual: usize },
    TooManySnapshots { max: usize, actual: usize },
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExperimentError::InvalidSampleEvery => write!(f, "sample_every must be positive"),
            ExperimentError::TooManySteps { max, actual } => {
                write!(f, "steps ({actual}) exceed supported maximum ({max})")
            }
            ExperimentError::TooManySamples { max, actual } => {
                write!(
                    f,
                    "sample count ({actual}) exceeds supported maximum ({max})"
                )
            }
            ExperimentError::TooManySnapshots { max, actual } => {
                write!(
                    f,
                    "snapshot count ({actual}) exceeds supported maximum ({max})"
                )
            }
        }
    }
}

impl Error for ExperimentError {}

impl World {
    pub const MAX_WORLD_SIZE: f64 = crate::constants::MAX_WORLD_SIZE;

    pub const MAX_EXPERIMENT_STEPS: usize = 1_000_000;
    pub const MAX_EXPERIMENT_SAMPLES: usize = 50_000;
    pub const MAX_EXPERIMENT_SNAPSHOTS: usize = 1_000;

    pub fn new(
        agents: Vec<Agent>,
        nns: Vec<NeuralNet>,
        config: SimConfig,
    ) -> Result<Self, WorldInitError> {
        config.validate()?;
        if config.num_organisms != nns.len() {
            return Err(WorldInitError::NumOrganismsMismatch {
                expected: config.num_organisms,
                actual: nns.len(),
            });
        }
        let expected_agent_count = config
            .num_organisms
            .checked_mul(config.agents_per_organism)
            .ok_or(WorldInitError::AgentCountOverflow)?;
        if expected_agent_count > SimConfig::MAX_TOTAL_AGENTS {
            return Err(WorldInitError::TooManyAgents {
                max: SimConfig::MAX_TOTAL_AGENTS,
                actual: expected_agent_count,
            });
        }
        if agents.len() != expected_agent_count {
            return Err(WorldInitError::AgentCountMismatch {
                expected: expected_agent_count,
                actual: agents.len(),
            });
        }
        if !agents.iter().all(|a| (a.organism_id as usize) < nns.len()) {
            return Err(WorldInitError::InvalidOrganismId);
        }

        // In Mode B (families non-empty) validate family count fits in u16 and
        // that the sum of initial_counts matches num_organisms exactly.
        if !config.families.is_empty() {
            if config.families.len() > u16::MAX as usize {
                return Err(WorldInitError::TooManyFamilies {
                    actual: config.families.len(),
                });
            }
            let family_total: usize = config
                .families
                .iter()
                .try_fold(0usize, |acc, f| acc.checked_add(f.initial_count))
                .ok_or(WorldInitError::AgentCountOverflow)?;
            if family_total != config.num_organisms {
                return Err(WorldInitError::FamilyOrganismCountMismatch {
                    expected: family_total,
                    actual: config.num_organisms,
                });
            }
        }

        let mut organisms: Vec<OrganismRuntime> = nns
            .into_iter()
            .enumerate()
            .map(|(id, nn)| {
                let genome = Genome::with_nn_weights(nn.to_weight_vec());
                let developmental_program = DevelopmentalProgram::decode(genome.segment_data(3));
                OrganismRuntime {
                    id: id as u16,
                    stable_id: id as u64,
                    generation: 0,
                    age_steps: 0,
                    alive: true,
                    boundary_integrity: 1.0,
                    metabolic_state: MetabolicState::default(),
                    genome: genome.clone(),
                    ancestor_genome: genome,
                    nn,
                    agent_ids: Vec::new(),
                    maturity: 1.0,
                    metabolism_engine: None,
                    developmental_program,
                    parent_stable_id: None,
                    family_id: 0,
                }
            })
            .collect();

        // Mode B: assign family_id to each organism based on families[*].initial_count ranges.
        if !config.families.is_empty() {
            let mut org_offset = 0usize;
            for (fid, family) in config.families.iter().enumerate() {
                for org in organisms
                    .iter_mut()
                    .skip(org_offset)
                    .take(family.initial_count)
                {
                    org.family_id = fid as u16;
                }
                org_offset += family.initial_count;
            }
        }

        for agent in &agents {
            organisms[agent.organism_id as usize]
                .agent_ids
                .push(agent.id);
        }

        // Graph mode: initialize each organism's metabolic genome segment with
        // small random values, then decode into per-organism metabolism engines.
        let mut init_rng = ChaCha12Rng::seed_from_u64(config.seed.wrapping_add(1));
        if config.metabolism_mode == MetabolismMode::Graph {
            for org in &mut organisms {
                let mut seg = [0.0f32; Genome::METABOLIC_SIZE];
                for v in &mut seg {
                    *v = init_rng.random_range(-0.5f32..0.5);
                }
                org.genome.set_segment_data(1, &seg);
                org.metabolism_engine =
                    decode_organism_metabolism(&org.genome, config.metabolism_mode);
            }
        }

        let max_agent_id = agents.iter().map(|a| a.id).max().unwrap_or(0);
        let metabolism = match config.metabolism_mode {
            MetabolismMode::Toy => MetabolismEngine::default(),
            MetabolismMode::Counter => {
                MetabolismEngine::Counter(crate::metabolism::CounterMetabolism::default())
            }
            MetabolismMode::Graph => {
                MetabolismEngine::Graph(crate::metabolism::GraphMetabolism::default())
            }
        };

        let world_size = config.world_size;
        let org_count = organisms.len();
        let agent_count = agents.len();
        let next_organism_stable_id = org_count as u64;

        // E5: build patchy resource field when patch_count > 0; use a separate
        // seeded RNG (seed+2) so the patch layout is deterministic per seed and
        // independent of the organism RNG (seed) and graph-metabolism RNG (seed+1).
        let resource_field = if config.resource_patch_count > 0 {
            let mut patch_rng = ChaCha12Rng::seed_from_u64(config.seed.wrapping_add(2));
            ResourceField::new_with_patches(
                world_size,
                1.0,
                1.0,
                config.resource_patch_count,
                config.resource_patch_scale,
                &mut patch_rng,
            )
        } else {
            ResourceField::new(world_size, 1.0, 1.0)
        };

        Ok(Self {
            agents,
            organisms,
            config: config.clone(),
            metabolism,
            resource_field,
            org_toroidal_sums: vec![[0.0, 0.0, 0.0, 0.0]; org_count],
            org_counts: vec![0; org_count],
            rng: ChaCha12Rng::seed_from_u64(config.seed),
            next_agent_id: max_agent_id.saturating_add(1),
            step_index: 0,
            original_config: None,
            scheduled_ablation_applied: false,
            births_last_step: 0,
            deaths_last_step: 0,
            births_per_family: vec![0; config.families.len()],
            deaths_per_family: vec![0; config.families.len()],
            total_births: 0,
            total_deaths: 0,
            mutation_rates: Self::mutation_rates_from_config(&config),
            next_organism_stable_id,
            agent_id_exhaustions_last_step: 0,
            total_agent_id_exhaustions: 0,
            lifespans: Vec::new(),
            lineage_events: Vec::new(),
            current_resource_rate: config.resource_regeneration_rate,
            deltas_buffer: Vec::with_capacity(agent_count),
            neighbor_sums_buffer: Vec::with_capacity(org_count),
            neighbor_counts_buffer: Vec::with_capacity(org_count),
            homeostasis_sums_buffer: Vec::with_capacity(org_count),
            homeostasis_counts_buffer: Vec::with_capacity(org_count),
        })
    }

    fn mutation_rates_from_config(config: &SimConfig) -> MutationRates {
        MutationRates {
            point_rate: config.mutation_point_rate,
            point_scale: config.mutation_point_scale,
            reset_rate: config.mutation_reset_rate,
            scale_rate: config.mutation_scale_rate,
            scale_min: config.mutation_scale_min,
            scale_max: config.mutation_scale_max,
            value_limit: config.mutation_value_limit,
        }
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: SimConfig) -> Result<(), WorldInitError> {
        let mode_changed = self.config.metabolism_mode != config.metabolism_mode;
        config.validate()?;
        if config.num_organisms != self.organisms.len() {
            return Err(WorldInitError::NumOrganismsMismatch {
                expected: config.num_organisms,
                actual: self.organisms.len(),
            });
        }
        // Reject family changes: organisms carry family_id indices that would become
        // stale if the families vec changed at runtime. Callers must create a new World
        // to change family layout.
        if config.families.len() != self.config.families.len() {
            return Err(WorldInitError::FamilyOrganismCountMismatch {
                expected: self.config.families.iter().map(|f| f.initial_count).sum(),
                actual: config.num_organisms,
            });
        }
        if !config.families.is_empty() {
            let family_total: usize = config
                .families
                .iter()
                .try_fold(0usize, |acc, f| acc.checked_add(f.initial_count))
                .ok_or(WorldInitError::AgentCountOverflow)?;
            if family_total != config.num_organisms {
                return Err(WorldInitError::FamilyOrganismCountMismatch {
                    expected: family_total,
                    actual: config.num_organisms,
                });
            }
        }
        let expected_agent_count = config
            .num_organisms
            .checked_mul(config.agents_per_organism)
            .ok_or(WorldInitError::AgentCountOverflow)?;
        if self.agents.len() != expected_agent_count {
            return Err(WorldInitError::AgentCountMismatch {
                expected: expected_agent_count,
                actual: self.agents.len(),
            });
        }
        let world_size_changed = (self.config.world_size - config.world_size).abs() > f64::EPSILON;
        let patch_params_changed = self.config.resource_patch_count != config.resource_patch_count
            || (self.config.resource_patch_scale - config.resource_patch_scale).abs()
                > f32::EPSILON
            || self.config.seed != config.seed;
        if world_size_changed || patch_params_changed {
            self.resource_field = if config.resource_patch_count > 0 {
                let mut patch_rng = ChaCha12Rng::seed_from_u64(config.seed.wrapping_add(2));
                ResourceField::new_with_patches(
                    config.world_size,
                    1.0,
                    1.0,
                    config.resource_patch_count,
                    config.resource_patch_scale,
                    &mut patch_rng,
                )
            } else {
                ResourceField::new(config.world_size, 1.0, 1.0)
            };
        }
        self.current_resource_rate = config.resource_regeneration_rate;
        self.config = config;
        self.original_config = None;
        self.scheduled_ablation_applied = false;
        self.mutation_rates = Self::mutation_rates_from_config(&self.config);
        if mode_changed {
            self.metabolism = match self.config.metabolism_mode {
                MetabolismMode::Toy => MetabolismEngine::default(),
                MetabolismMode::Counter => {
                    MetabolismEngine::Counter(crate::metabolism::CounterMetabolism::default())
                }
                MetabolismMode::Graph => {
                    MetabolismEngine::Graph(crate::metabolism::GraphMetabolism::default())
                }
            };
            for org in &mut self.organisms {
                org.metabolism_engine =
                    decode_organism_metabolism(&org.genome, self.config.metabolism_mode);
            }
        }
        Ok(())
    }

    pub fn set_metabolism_engine(&mut self, engine: MetabolismEngine) {
        self.metabolism = engine;
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    pub fn resource_field(&self) -> &ResourceField {
        &self.resource_field
    }

    pub fn resource_field_mut(&mut self) -> &mut ResourceField {
        &mut self.resource_field
    }

    pub fn metabolic_state(&self, organism_id: usize) -> Option<&MetabolicState> {
        self.organisms.get(organism_id).map(|o| &o.metabolic_state)
    }

    pub fn organism_count(&self) -> usize {
        self.organisms.len()
    }

    pub fn population_stats(&self) -> PopulationStats {
        let alive = self.alive_count();
        let generation_sum = self
            .organisms
            .iter()
            .filter(|o| o.alive)
            .map(|o| o.generation as f32)
            .sum::<f32>();
        PopulationStats {
            population_size: alive,
            alive_count: alive,
            total_births: self.total_births,
            total_deaths: self.total_deaths,
            mean_generation: if alive > 0 {
                generation_sum / alive as f32
            } else {
                0.0
            },
        }
    }

    fn live_flags(&self) -> Vec<bool> {
        self.organisms.iter().map(|o| o.alive).collect()
    }

    fn alive_count(&self) -> usize {
        self.organisms.iter().filter(|o| o.alive).count()
    }

    fn terminal_boundary_threshold(&self) -> f32 {
        self.config
            .boundary_collapse_threshold
            .max(self.config.death_boundary_threshold)
    }

    /// Returns the effective enable-flag for a given capability, respecting per-family config.
    ///
    /// - Mode A (`families` is empty): returns `global` unchanged.
    /// - Mode B (`families` is non-empty): returns the per-family selector result.
    ///   Global flags are intentionally ignored in Mode B; per-family `enable_*` fields are
    ///   the authoritative ablation mechanism. Scheduled ablation in Mode B should target
    ///   `FamilyConfig` fields directly.
    ///
    /// An out-of-range `family_id` is a programming error: it panics in debug builds
    /// (via `debug_assert`) and fails closed (`false`) in release to avoid enabling
    /// a criterion that the family never explicitly opted into.
    #[inline]
    pub(crate) fn family_flag(
        families: &[FamilyConfig],
        family_id: u16,
        selector: impl Fn(&FamilyConfig) -> bool,
        global: bool,
    ) -> bool {
        if families.is_empty() {
            global
        } else {
            debug_assert!(
                (family_id as usize) < families.len(),
                "family_id {family_id} is out of range for families len {}",
                families.len()
            );
            // Fail-closed: unknown family_id → false (criterion disabled).
            families.get(family_id as usize).is_some_and(selector)
        }
    }

    fn next_agent_id_checked(&mut self) -> Option<u32> {
        if self.next_agent_id == u32::MAX {
            return None;
        }
        let id = self.next_agent_id;
        self.next_agent_id += 1;
        Some(id)
    }

    fn compute_organism_centers_with_counts(&self) -> (Vec<Option<[f64; 2]>>, Vec<usize>) {
        let world_size = self.config.world_size;
        let tau_over_world = (2.0 * PI) / world_size;
        let mut sums = vec![[0.0f64, 0.0, 0.0, 0.0]; self.organisms.len()];
        let mut counts = vec![0usize; self.organisms.len()];

        for agent in &self.agents {
            let idx = agent.organism_id as usize;
            if !self.organisms.get(idx).map(|o| o.alive).unwrap_or(false) {
                continue;
            }
            let theta_x = agent.position[0] * tau_over_world;
            let theta_y = agent.position[1] * tau_over_world;
            let (sin_x, cos_x) = theta_x.sin_cos();
            let (sin_y, cos_y) = theta_y.sin_cos();
            sums[idx][0] += sin_x;
            sums[idx][1] += cos_x;
            sums[idx][2] += sin_y;
            sums[idx][3] += cos_y;
            counts[idx] += 1;
        }

        let mut centers = vec![None; self.organisms.len()];
        for idx in 0..self.organisms.len() {
            if counts[idx] == 0 {
                continue;
            }
            centers[idx] = Some([
                Self::toroidal_mean_coord(sums[idx][0], sums[idx][1], world_size),
                Self::toroidal_mean_coord(sums[idx][2], sums[idx][3], world_size),
            ]);
        }
        (centers, counts)
    }

    fn compute_organism_centers(&self) -> Vec<Option<[f64; 2]>> {
        self.compute_organism_centers_with_counts().0
    }

    fn prune_dead_entities(&mut self) {
        if self.organisms.iter().all(|o| o.alive) {
            return;
        }

        let old_organisms = std::mem::take(&mut self.organisms);
        let mut remap = vec![None::<u16>; old_organisms.len()];
        let mut new_organisms = Vec::with_capacity(old_organisms.len());
        for (old_idx, mut org) in old_organisms.into_iter().enumerate() {
            if !org.alive {
                continue;
            }
            let new_id = new_organisms.len() as u16;
            remap[old_idx] = Some(new_id);
            org.id = new_id;
            org.agent_ids.clear();
            new_organisms.push(org);
        }

        let old_agents = std::mem::take(&mut self.agents);
        let mut new_agents = Vec::with_capacity(old_agents.len());
        for mut agent in old_agents {
            if let Some(new_org_id) = remap
                .get(agent.organism_id as usize)
                .and_then(|mapped| *mapped)
            {
                agent.organism_id = new_org_id;
                new_agents.push(agent);
            }
        }

        self.organisms = new_organisms;
        self.agents = new_agents;
        for agent in &self.agents {
            self.organisms[agent.organism_id as usize]
                .agent_ids
                .push(agent.id);
        }
        self.org_toroidal_sums
            .resize(self.organisms.len(), [0.0, 0.0, 0.0, 0.0]);
        self.org_counts.resize(self.organisms.len(), 0);
        self.org_toroidal_sums.fill([0.0, 0.0, 0.0, 0.0]);
        self.org_counts.fill(0);
    }

    fn toroidal_mean_coord(sum_sin: f64, sum_cos: f64, world_size: f64) -> f64 {
        if sum_sin == 0.0 && sum_cos == 0.0 {
            return 0.0;
        }
        let angle = sum_sin.atan2(sum_cos);
        (angle.rem_euclid(2.0 * PI) / (2.0 * PI)) * world_size
    }

    /// Effective sensing radius for an organism, accounting for developmental stage.
    fn effective_sensing_radius(&self, org_idx: usize) -> f64 {
        let dev_sensing = if self.config.enable_growth {
            self.organisms[org_idx]
                .developmental_program
                .stage_factors(self.organisms[org_idx].maturity)
                .1
        } else {
            1.0
        };
        self.config.sensing_radius * dev_sensing as f64
    }

    pub fn run_experiment(&mut self, steps: usize, sample_every: usize) -> RunSummary {
        self.try_run_experiment(steps, sample_every)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_run_experiment(
        &mut self,
        steps: usize,
        sample_every: usize,
    ) -> Result<RunSummary, ExperimentError> {
        if sample_every == 0 {
            return Err(ExperimentError::InvalidSampleEvery);
        }
        if steps > Self::MAX_EXPERIMENT_STEPS {
            return Err(ExperimentError::TooManySteps {
                max: Self::MAX_EXPERIMENT_STEPS,
                actual: steps,
            });
        }
        let estimated_samples = if steps == 0 {
            0
        } else {
            ((steps - 1) / sample_every) + 1
        };
        if estimated_samples > Self::MAX_EXPERIMENT_SAMPLES {
            return Err(ExperimentError::TooManySamples {
                max: Self::MAX_EXPERIMENT_SAMPLES,
                actual: estimated_samples,
            });
        }

        self.lifespans.clear();
        self.lineage_events.clear();
        let births_before = self.total_births;
        let mut samples = Vec::with_capacity(estimated_samples);
        for step in 1..=steps {
            self.step();
            if step % sample_every == 0 || step == steps {
                let mut metrics = crate::metrics::collect_step_metrics(
                    step,
                    self.step_index,
                    self.config.world_size,
                    self.resource_field.total(),
                    self.births_last_step,
                    self.deaths_last_step,
                    self.agent_id_exhaustions_last_step,
                    &self.organisms,
                    &self.agents,
                );
                if !self.config.families.is_empty() {
                    for fid in 0..self.config.families.len() {
                        metrics
                            .family_breakdown
                            .push(crate::metrics::collect_family_metrics(
                                fid as u16,
                                self.step_index,
                                self.births_per_family.get(fid).copied().unwrap_or(0),
                                self.deaths_per_family.get(fid).copied().unwrap_or(0),
                                &self.organisms,
                            ));
                    }
                }
                samples.push(metrics);
            }
        }
        Ok(RunSummary {
            schema_version: 1,
            steps,
            sample_every,
            final_alive_count: self.alive_count(),
            samples,
            lifespans: std::mem::take(&mut self.lifespans),
            total_reproduction_events: self.total_births - births_before,
            lineage_events: std::mem::take(&mut self.lineage_events),
            organism_snapshots: Vec::new(),
        })
    }

    /// Collect a snapshot of all alive organisms at the current step.
    ///
    /// Computes centers and agent counts directly so snapshot correctness does
    /// not depend on whether metabolism is enabled this step.
    fn collect_organism_snapshots(&self, step: usize) -> SnapshotFrame {
        let (centers, counts) = self.compute_organism_centers_with_counts();
        let organisms: Vec<OrganismSnapshot> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, org)| org.alive)
            .map(|(idx, org)| {
                let center = centers.get(idx).and_then(|c| *c).unwrap_or([0.0, 0.0]);
                OrganismSnapshot {
                    stable_id: org.stable_id,
                    generation: org.generation,
                    age_steps: org.age_steps,
                    energy: org.metabolic_state.energy,
                    waste: org.metabolic_state.waste,
                    boundary_integrity: org.boundary_integrity,
                    maturity: org.maturity,
                    center_x: center[0],
                    center_y: center[1],
                    n_agents: counts[idx],
                }
            })
            .collect();
        SnapshotFrame { step, organisms }
    }

    /// Run an experiment like `try_run_experiment`, but also collect per-organism
    /// snapshots at the specified steps.
    pub fn try_run_experiment_with_snapshots(
        &mut self,
        steps: usize,
        sample_every: usize,
        snapshot_steps: &[usize],
    ) -> Result<RunSummary, ExperimentError> {
        if sample_every == 0 {
            return Err(ExperimentError::InvalidSampleEvery);
        }
        if steps > Self::MAX_EXPERIMENT_STEPS {
            return Err(ExperimentError::TooManySteps {
                max: Self::MAX_EXPERIMENT_STEPS,
                actual: steps,
            });
        }
        if snapshot_steps.len() > Self::MAX_EXPERIMENT_SNAPSHOTS {
            return Err(ExperimentError::TooManySnapshots {
                max: Self::MAX_EXPERIMENT_SNAPSHOTS,
                actual: snapshot_steps.len(),
            });
        }
        let estimated_samples = if steps == 0 {
            0
        } else {
            ((steps - 1) / sample_every) + 1
        };
        if estimated_samples > Self::MAX_EXPERIMENT_SAMPLES {
            return Err(ExperimentError::TooManySamples {
                max: Self::MAX_EXPERIMENT_SAMPLES,
                actual: estimated_samples,
            });
        }

        self.lifespans.clear();
        self.lineage_events.clear();
        let births_before = self.total_births;
        let mut samples = Vec::with_capacity(estimated_samples);
        let mut snapshots = Vec::with_capacity(snapshot_steps.len());
        let snapshot_steps_set: HashSet<usize> = snapshot_steps.iter().copied().collect();

        for step in 1..=steps {
            self.step();
            if step % sample_every == 0 || step == steps {
                let mut metrics = crate::metrics::collect_step_metrics(
                    step,
                    self.step_index,
                    self.config.world_size,
                    self.resource_field.total(),
                    self.births_last_step,
                    self.deaths_last_step,
                    self.agent_id_exhaustions_last_step,
                    &self.organisms,
                    &self.agents,
                );
                if !self.config.families.is_empty() {
                    for fid in 0..self.config.families.len() {
                        metrics
                            .family_breakdown
                            .push(crate::metrics::collect_family_metrics(
                                fid as u16,
                                self.step_index,
                                self.births_per_family.get(fid).copied().unwrap_or(0),
                                self.deaths_per_family.get(fid).copied().unwrap_or(0),
                                &self.organisms,
                            ));
                    }
                }
                samples.push(metrics);
            }
            if snapshot_steps_set.contains(&step) {
                snapshots.push(self.collect_organism_snapshots(step));
            }
        }
        Ok(RunSummary {
            schema_version: 1,
            steps,
            sample_every,
            final_alive_count: self.alive_count(),
            samples,
            lifespans: std::mem::take(&mut self.lifespans),
            total_reproduction_events: self.total_births - births_before,
            lineage_events: std::mem::take(&mut self.lineage_events),
            organism_snapshots: snapshots,
        })
    }

    fn mark_dead(&mut self, org_idx: usize) {
        if let Some(org) = self.organisms.get_mut(org_idx) {
            if org.alive {
                self.lifespans.push(org.age_steps);
                let family_id = org.family_id;
                org.alive = false;
                org.boundary_integrity = 0.0;
                self.deaths_last_step += 1;
                self.total_deaths += 1;
                if let Some(cnt) = self.deaths_per_family.get_mut(family_id as usize) {
                    *cnt += 1;
                }
            }
        }
    }

    fn maybe_reproduce(&mut self) {
        let child_agents =
            (self.config.agents_per_organism / 2).max(self.config.reproduction_child_min_agents);
        let parent_indices: Vec<usize> = self
            .organisms
            .iter()
            .enumerate()
            .filter_map(|(idx, org)| {
                let mature_enough = org.maturity >= 1.0;
                let can_reproduce = Self::family_flag(
                    &self.config.families,
                    org.family_id,
                    |f| f.enable_reproduction,
                    self.config.enable_reproduction,
                );
                (org.alive
                    && can_reproduce
                    && org.metabolic_state.energy >= self.config.reproduction_min_energy
                    && org.boundary_integrity >= self.config.reproduction_min_boundary
                    && mature_enough)
                    .then_some(idx)
            })
            .collect();
        if parent_indices.is_empty() {
            return;
        }
        let centers = self.compute_organism_centers();

        for parent_idx in parent_indices {
            if self
                .agents
                .len()
                .checked_add(child_agents)
                .map(|n| n > SimConfig::MAX_TOTAL_AGENTS)
                .unwrap_or(true)
            {
                break;
            }
            let remaining_ids = u32::MAX as u64 - self.next_agent_id as u64;
            if remaining_ids + 1 < child_agents as u64 {
                self.agent_id_exhaustions_last_step += 1;
                self.total_agent_id_exhaustions += 1;
                break;
            }

            let child_id = match u16::try_from(self.organisms.len()) {
                Ok(id) => id,
                Err(_) => break,
            };

            let center = centers
                .get(parent_idx)
                .and_then(|c| *c)
                .unwrap_or([0.0, 0.0]);

            self.spawn_child(parent_idx, child_id, center, child_agents);
        }
    }

    fn spawn_child(
        &mut self,
        parent_idx: usize,
        child_id: u16,
        center: [f64; 2],
        child_agents: usize,
    ) {
        let (
            parent_generation,
            parent_stable_id,
            parent_ancestor,
            mut child_genome,
            parent_family_id,
        ) = {
            let parent = &self.organisms[parent_idx];
            if !parent.alive || parent.metabolic_state.energy < self.config.reproduction_energy_cost
            {
                return;
            }
            (
                parent.generation,
                parent.stable_id,
                parent.ancestor_genome.clone(),
                parent.genome.clone(),
                parent.family_id,
            )
        };

        let evolve = Self::family_flag(
            &self.config.families,
            parent_family_id,
            |f| f.enable_evolution,
            self.config.enable_evolution,
        );
        if evolve {
            // Apply per-family mutation_rate_multiplier when families are defined.
            // Only the selection-probability rates are scaled; mutation magnitudes
            // (point_scale, scale_min/max, value_limit) are unchanged by design —
            // mutation_rate_multiplier controls how often mutations fire, not how large.
            let effective_rates = if !self.config.families.is_empty() {
                let multiplier = self
                    .config
                    .families
                    .get(parent_family_id as usize)
                    .map(|f| f.mutation_rate_multiplier)
                    .unwrap_or(1.0); // out-of-range family_id: no-op multiplier (neutral)
                crate::genome::MutationRates {
                    // Clamp to 1.0: rates are probabilities and must stay in [0, 1].
                    point_rate: (self.mutation_rates.point_rate * multiplier).min(1.0),
                    reset_rate: (self.mutation_rates.reset_rate * multiplier).min(1.0),
                    scale_rate: (self.mutation_rates.scale_rate * multiplier).min(1.0),
                    ..self.mutation_rates
                }
            } else {
                self.mutation_rates
            };
            child_genome.mutate(&mut self.rng, &effective_rates);
        }
        let child_weights = if child_genome.nn_weights().len() == NeuralNet::WEIGHT_COUNT {
            child_genome.nn_weights().to_vec()
        } else {
            self.organisms[parent_idx].nn.to_weight_vec()
        };
        let child_nn = NeuralNet::from_weights(child_weights.into_iter());
        let mut child_agent_ids = Vec::with_capacity(child_agents);

        for _ in 0..child_agents {
            let theta = self.rng.random::<f64>() * 2.0 * PI;
            let radius = self.rng.random::<f64>().sqrt() * self.config.reproduction_spawn_radius;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let pos = [
                (center[0] + radius * cos_theta).rem_euclid(self.config.world_size),
                (center[1] + radius * sin_theta).rem_euclid(self.config.world_size),
            ];
            let Some(id) = self.next_agent_id_checked() else {
                break;
            };
            let mut agent = Agent::new(id, child_id, pos);
            agent.internal_state[2] = 1.0;
            child_agent_ids.push(id);
            self.agents.push(agent);
        }
        if child_agent_ids.is_empty() {
            return;
        }

        self.organisms[parent_idx].metabolic_state.energy -= self.config.reproduction_energy_cost;

        let metabolic_state = MetabolicState {
            energy: self.config.reproduction_energy_cost,
            ..MetabolicState::default()
        };
        let child_metabolism_engine =
            decode_organism_metabolism(&child_genome, self.config.metabolism_mode);
        let developmental_program = DevelopmentalProgram::decode(child_genome.segment_data(3));
        let child_stable_id = self.next_organism_stable_id;
        let child_generation = parent_generation + 1;
        let child = OrganismRuntime {
            id: child_id,
            stable_id: child_stable_id,
            generation: child_generation,
            age_steps: 0,
            alive: true,
            boundary_integrity: 1.0,
            metabolic_state,
            genome: child_genome,
            ancestor_genome: parent_ancestor,
            nn: child_nn,
            agent_ids: child_agent_ids,
            maturity: 0.0,
            metabolism_engine: child_metabolism_engine,
            developmental_program,
            parent_stable_id: Some(parent_stable_id),
            family_id: parent_family_id,
        };
        self.next_organism_stable_id = self.next_organism_stable_id.saturating_add(1);
        let fnv1a = |data: &[f32]| -> u64 {
            const OFFSET: u64 = 0xcbf29ce484222325;
            const PRIME: u64 = 0x100000001b3;
            data.iter()
                .flat_map(|f| f.to_le_bytes())
                .fold(OFFSET, |h, b| (h ^ b as u64).wrapping_mul(PRIME))
        };
        let parent_genome_hash = fnv1a(self.organisms[parent_idx].genome.data());
        let genome_hash = fnv1a(child.genome.data());
        let parent_child_genome_distance = {
            let parent_data = self.organisms[parent_idx].genome.data();
            let child_data = child.genome.data();
            assert_eq!(
                parent_data.len(),
                child_data.len(),
                "parent and child genome lengths must match"
            );
            let n = parent_data.len();
            if n == 0 {
                0.0f32
            } else {
                let sum_sq: f32 = parent_data
                    .iter()
                    .zip(child_data.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum();
                (sum_sq / n as f32).sqrt()
            }
        };
        self.lineage_events.push(LineageEvent {
            step: self.step_index,
            parent_stable_id,
            child_stable_id,
            generation: child_generation,
            genome_hash,
            family_id: parent_family_id,
            parent_genome_hash,
            parent_child_genome_distance,
        });
        self.organisms.push(child);
        self.org_toroidal_sums.push([0.0, 0.0, 0.0, 0.0]);
        self.org_counts.push(0);
        self.births_last_step += 1;
        self.total_births += 1;
        if let Some(cnt) = self.births_per_family.get_mut(parent_family_id as usize) {
            *cnt += 1;
        }
    }

    fn apply_scheduled_ablation_if_due(&mut self) {
        if self.scheduled_ablation_applied {
            return;
        }
        if self.config.ablation_step == 0 || self.step_index < self.config.ablation_step {
            return;
        }
        if self.original_config.is_none() {
            self.original_config = Some(self.config.clone());
        }
        for target in &self.config.ablation_targets {
            match target {
                AblationTarget::Metabolism => self.config.enable_metabolism = false,
                AblationTarget::Boundary => self.config.enable_boundary_maintenance = false,
                AblationTarget::Homeostasis => self.config.enable_homeostasis = false,
                AblationTarget::Response => self.config.enable_response = false,
                AblationTarget::Reproduction => self.config.enable_reproduction = false,
                AblationTarget::Evolution => self.config.enable_evolution = false,
                AblationTarget::Growth => self.config.enable_growth = false,
            }
        }
        self.scheduled_ablation_applied = true;
    }

    pub fn step(&mut self) -> StepTimings {
        let total_start = Instant::now();
        self.step_index = self.step_index.saturating_add(1);
        self.apply_scheduled_ablation_if_due();
        self.births_last_step = 0;
        self.deaths_last_step = 0;
        self.births_per_family.fill(0);
        self.deaths_per_family.fill(0);
        self.agent_id_exhaustions_last_step = 0;
        let boundary_terminal_threshold = self.terminal_boundary_threshold();

        let t0 = Instant::now();
        let live_flags = self.live_flags();
        let tree = spatial::build_index_active(&self.agents, &live_flags);
        let spatial_build_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        self.step_nn_query_phase(&tree);
        let nn_query_us = t1.elapsed().as_micros() as u64;

        let t2 = Instant::now();
        self.step_agent_state_phase();
        self.step_boundary_phase(boundary_terminal_threshold);
        self.step_metabolism_phase(boundary_terminal_threshold);
        self.step_growth_and_crowding_phase(boundary_terminal_threshold);

        // In Mode B each family controls its own enable_reproduction flag;
        // we enter maybe_reproduce() when at least one family allows it.
        // maybe_reproduce() filters per organism via family_flag, so families with
        // enable_reproduction=false are silently skipped inside.
        if self.config.enable_reproduction
            || self.config.families.iter().any(|f| f.enable_reproduction)
        {
            self.maybe_reproduce();
        }
        let dead_count = self.organisms.iter().filter(|o| !o.alive).count();
        if dead_count > 0
            && (self
                .step_index
                .checked_rem(self.config.compaction_interval_steps)
                .is_some_and(|r| r == 0)
                || dead_count * 4 >= self.organisms.len().max(1))
        {
            self.prune_dead_entities();
        }

        self.step_environment_phase(&tree);

        let state_update_us = t2.elapsed().as_micros() as u64;

        StepTimings {
            spatial_build_us,
            nn_query_us,
            state_update_us,
            total_us: total_start.elapsed().as_micros() as u64,
        }
    }
}

mod phases;
#[cfg(test)]
mod tests;
