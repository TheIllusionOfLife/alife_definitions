use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    pub seed: u64,
    pub world_size: f64,
    pub num_organisms: usize,
    pub agents_per_organism: usize,
    pub sensing_radius: f64,
    pub max_speed: f64,
    pub dt: f64,
    pub neighbor_norm: f64,
    pub enable_metabolism: bool,
    pub enable_boundary_maintenance: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            world_size: 100.0,
            num_organisms: 50,
            agents_per_organism: 50,
            sensing_radius: 5.0,
            max_speed: 2.0,
            dt: 0.1,
            neighbor_norm: 50.0,
            enable_metabolism: true,
            enable_boundary_maintenance: true,
        }
    }
}
