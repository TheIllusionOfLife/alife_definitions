/// 2D grid resource field stub.
/// Each cell holds a resource concentration value.

#[derive(Clone, Debug)]
pub struct ResourceField {
    width: usize,
    height: usize,
    cell_size: f64,
    data: Vec<f32>,
    total: f64,
    initial_value: f32,
    /// Per-cell regeneration rate multiplier for E5 spatial patchiness.
    /// Empty = uniform (all cells use the global rate as-is).
    /// When non-empty, `regenerate(rate)` uses `rate * multiplier[i]` per cell.
    /// Invariant: mean(rate_multiplier) ≈ 1.0 when non-empty (global budget preserved).
    rate_multiplier: Vec<f32>,
}

impl ResourceField {
    pub fn new(world_size: f64, cell_size: f64, initial_value: f32) -> Self {
        assert!(world_size > 0.0, "world_size must be positive");
        assert!(cell_size > 0.0, "cell_size must be positive");
        // The simulation world is currently square; use a square resource grid for parity.
        let width = (world_size / cell_size).ceil() as usize;
        let height = width;
        let data = vec![initial_value; width * height];
        let total = initial_value as f64 * (width * height) as f64;
        Self {
            width,
            height,
            cell_size,
            data,
            total,
            initial_value,
            rate_multiplier: Vec::new(),
        }
    }

    /// Build a resource field with spatially patchy regeneration rates (E5 regime).
    ///
    /// `patch_count` centres are placed uniformly at random.  Each cell's regeneration
    /// multiplier is the Gaussian falloff to the nearest patch centre (toroidal distance).
    /// The multipliers are normalized so `mean == 1.0`, preserving global resource budget.
    ///
    /// When `patch_count == 0` the field is identical to `ResourceField::new()` (uniform).
    pub fn new_with_patches(
        world_size: f64,
        cell_size: f64,
        initial_value: f32,
        patch_count: usize,
        patch_scale: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let mut field = Self::new(world_size, cell_size, initial_value);
        if patch_count == 0 || (patch_scale - 1.0).abs() < f32::EPSILON {
            return field; // uniform — rate_multiplier stays empty
        }

        let n_cells = field.width * field.height;
        let ws = world_size as f32;
        // sigma: half the average inter-patch spacing so patches blend without dead zones
        let sigma = ws / (2.0 * (patch_count as f32).sqrt().max(1.0));

        // Random patch centres (world-space coordinates)
        let centers: Vec<(f32, f32)> = (0..patch_count)
            .map(|_| (rng.random::<f32>() * ws, rng.random::<f32>() * ws))
            .collect();

        let mut multipliers = vec![0.0f32; n_cells];
        for cy in 0..field.height {
            for cx in 0..field.width {
                let cell_x = (cx as f32 + 0.5) * cell_size as f32;
                let cell_y = (cy as f32 + 0.5) * cell_size as f32;

                // Maximum Gaussian falloff across all patch centres (toroidal distance)
                let max_falloff = centers
                    .iter()
                    .map(|&(px, py)| {
                        let dx = (cell_x - px).abs();
                        let dy = (cell_y - py).abs();
                        let dx = dx.min(ws - dx); // toroidal wrap
                        let dy = dy.min(ws - dy);
                        let dist2 = dx * dx + dy * dy;
                        (-dist2 / (2.0 * sigma * sigma)).exp()
                    })
                    .fold(0.0f32, f32::max);

                multipliers[cy * field.width + cx] = 1.0 + (patch_scale - 1.0) * max_falloff;
            }
        }

        // Normalize so mean(multiplier) == 1.0 (global resource budget invariant)
        let mean = multipliers.iter().sum::<f32>() / n_cells as f32;
        if mean > 0.0 {
            for m in &mut multipliers {
                *m /= mean;
            }
        }

        field.rate_multiplier = multipliers;
        field
    }

    /// Regenerate resources toward the initial value at the given rate per step.
    ///
    /// Cells are capped at `initial_value`; cells already at or above it are unchanged.
    /// When a `rate_multiplier` is set (E5 regime) each cell uses `rate * multiplier[i]`.
    pub fn regenerate(&mut self, rate: f32) {
        debug_assert!(rate >= 0.0, "regeneration rate cannot be negative");
        if self.rate_multiplier.is_empty() {
            for cell in &mut self.data {
                let before = *cell;
                *cell = (*cell + rate).min(self.initial_value);
                self.total += (*cell - before) as f64;
            }
        } else {
            for (i, cell) in self.data.iter_mut().enumerate() {
                let effective_rate = rate * self.rate_multiplier[i];
                let before = *cell;
                *cell = (*cell + effective_rate).min(self.initial_value);
                self.total += (*cell - before) as f64;
            }
        }
    }

    /// Read-only view of the per-cell regeneration multipliers (empty = uniform).
    pub fn rate_multiplier(&self) -> &[f32] {
        &self.rate_multiplier
    }

    /// Get resource value at position. Coordinates wrap toroidally.
    pub fn get(&self, x: f64, y: f64) -> f32 {
        let (cx, cy) = self.wrap_coords(x, y);
        self.data[cy * self.width + cx]
    }

    /// Set resource value at position. Coordinates wrap toroidally.
    pub fn set(&mut self, x: f64, y: f64, value: f32) {
        let (cx, cy) = self.wrap_coords(x, y);
        let idx = cy * self.width + cx;
        let old = self.data[idx];
        self.data[idx] = value;
        self.total += (value - old) as f64;
    }

    /// Remove up to `amount` resource from the addressed cell and return actual amount withdrawn.
    pub fn take(&mut self, x: f64, y: f64, amount: f32) -> f32 {
        let (cx, cy) = self.wrap_coords(x, y);
        let idx = cy * self.width + cx;
        let removed = self.data[idx].min(amount.max(0.0));
        self.data[idx] -= removed;
        self.total -= removed as f64;
        removed
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn total(&self) -> f64 {
        self.total
    }

    fn wrap_coords(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = ((x / self.cell_size).floor() as isize).rem_euclid(self.width as isize) as usize;
        let cy = ((y / self.cell_size).floor() as isize).rem_euclid(self.height as isize) as usize;
        (cx, cy)
    }

    #[allow(dead_code)]
    fn clamp_coords(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = ((x / self.cell_size).max(0.0) as usize).min(self.width - 1);
        let cy = ((y / self.cell_size).max(0.0) as usize).min(self.height - 1);
        (cx, cy)
    }
}

#[cfg(test)]
mod tests {
    use super::ResourceField;

    #[test]
    fn wraps_coordinates_toroidally() {
        let mut field = ResourceField::new(10.0, 1.0, 0.0);
        field.set(9.0, 9.0, 3.0);
        assert!((field.get(-1.0, -1.0) - 3.0).abs() < f32::EPSILON);
        assert!((field.get(19.0, 19.0) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn take_withdraws_and_clamps_to_available() {
        let mut field = ResourceField::new(10.0, 1.0, 0.0);
        field.set(2.0, 3.0, 1.5);
        assert!((field.take(2.0, 3.0, 0.5) - 0.5).abs() < f32::EPSILON);
        assert!((field.get(2.0, 3.0) - 1.0).abs() < f32::EPSILON);
        assert!((field.take(2.0, 3.0, 5.0) - 1.0).abs() < f32::EPSILON);
        assert!((field.get(2.0, 3.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn total_tracks_updates_and_withdrawals() {
        let mut field = ResourceField::new(10.0, 1.0, 1.0);
        let initial = field.total();
        field.set(0.0, 0.0, 2.0);
        assert!((field.total() - (initial + 1.0)).abs() < 1e-6);
        let _ = field.take(0.0, 0.0, 0.5);
        assert!((field.total() - (initial + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn regenerate_restores_depleted_cells() {
        let mut field = ResourceField::new(10.0, 1.0, 1.0);
        field.set(3.0, 3.0, 0.0);
        field.regenerate(0.25);
        assert!((field.get(3.0, 3.0) - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn regenerate_caps_at_initial_value() {
        let mut field = ResourceField::new(10.0, 1.0, 1.0);
        field.regenerate(0.5);
        // Already at 1.0 (initial), should stay at 1.0
        assert!((field.get(0.0, 0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn resource_total_tracks_regeneration() {
        let mut field = ResourceField::new(10.0, 1.0, 1.0);
        field.set(0.0, 0.0, 0.0);
        let before = field.total();
        field.regenerate(0.5);
        assert!(field.total() > before);
    }
}
