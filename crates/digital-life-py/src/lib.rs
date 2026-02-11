use digital_life_core::agent::Agent;
use digital_life_core::config::SimConfig;
use digital_life_core::nn::NeuralNet;
use digital_life_core::world::World;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Minimal PyO3 module exposing digital-life-core to Python.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn default_config_json() -> PyResult<String> {
    serde_json::to_string(&SimConfig::default())
        .map_err(|e| PyValueError::new_err(format!("failed to serialize default config: {e}")))
}

#[pyfunction]
fn validate_config_json(config_json: &str) -> PyResult<bool> {
    let config: SimConfig = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("invalid config json: {e}")))?;
    let (agents, nns) = bootstrap_entities(config.num_organisms, config.agents_per_organism);
    World::try_new(agents, nns, config)
        .map(|_| true)
        .map_err(|e| PyValueError::new_err(format!("invalid world configuration: {e}")))
}

#[pyfunction]
fn step_once(
    num_organisms: usize,
    agents_per_organism: usize,
    world_size: f64,
) -> PyResult<(usize, u64)> {
    let config = SimConfig {
        num_organisms,
        agents_per_organism,
        world_size,
        ..SimConfig::default()
    };

    let (agents, nns) = bootstrap_entities(num_organisms, agents_per_organism);
    let mut world = World::try_new(agents, nns, config)
        .map_err(|e| PyValueError::new_err(format!("invalid world configuration: {e}")))?;
    let timings = world.step();
    Ok((world.agents.len(), timings.total_us))
}

fn bootstrap_entities(
    num_organisms: usize,
    agents_per_organism: usize,
) -> (Vec<Agent>, Vec<NeuralNet>) {
    let total_agents = num_organisms * agents_per_organism;
    let agents = (0..total_agents)
        .map(|i| {
            let organism_id = (i / agents_per_organism.max(1)) as u16;
            Agent::new(i as u32, organism_id, [50.0, 50.0])
        })
        .collect();
    let nns = (0..num_organisms)
        .map(|_| NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT)))
        .collect();
    (agents, nns)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(step_once, m)?)?;
    Ok(())
}
