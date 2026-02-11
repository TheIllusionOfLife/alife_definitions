#[derive(Clone, Debug)]
pub struct MetabolicState {
    pub energy: f32,
    pub resource: f32,
    pub waste: f32,
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self {
            energy: 0.5,
            resource: 5.0,
            waste: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToyMetabolism {
    pub uptake_rate: f32,
    pub conversion_efficiency: f32,
    pub waste_ratio: f32,
    pub energy_loss_rate: f32,
}

impl Default for ToyMetabolism {
    fn default() -> Self {
        Self {
            uptake_rate: 0.4,
            conversion_efficiency: 0.8,
            waste_ratio: 0.2,
            energy_loss_rate: 0.02,
        }
    }
}

impl ToyMetabolism {
    pub fn step(&self, state: &mut MetabolicState, dt: f32) {
        let uptake = (self.uptake_rate * dt).min(state.resource).max(0.0);
        state.resource -= uptake;
        state.energy += uptake * self.conversion_efficiency;
        state.waste += uptake * self.waste_ratio;

        // Minimal thermodynamic loss to avoid unbounded free energy growth.
        let retained = (1.0 - self.energy_loss_rate * dt).clamp(0.0, 1.0);
        state.energy = (state.energy * retained).max(0.0);
    }
}
