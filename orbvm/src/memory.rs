//! Phase Memory - White's dispersion equation implementation

use ndarray::{Array1, Array2};
use num_complex::Complex64;

pub struct PhaseMemory {
    n_modes: usize,
    history_size: usize,
}

impl PhaseMemory {
    pub fn new(n_modes: usize) -> Self {
        Self {
            n_modes,
            history_size: 1000,
        }
    }

    pub fn solve_white_dispersion(
        &mut self,
        initial_state: &Array1<f64>,
        diffusion: f64,
        _potential: f64,
        dt: f64,
        steps: usize,
    ) -> Array1<f64> {
        let mut state = initial_state.clone();
        for _ in 0..steps {
            let mut laplacian = Array1::zeros(self.n_modes);
            for i in 1..self.n_modes - 1 {
                laplacian[i] = state[i - 1] - 2.0 * state[i] + state[i + 1];
            }
            state = &state + &(&laplacian * diffusion) * dt;
        }
        state
    }

    pub fn compute_lambda_2(&self, adjacency: &Array2<f64>) -> f64 {
        let n = adjacency.nrows() as f64;
        let sum: f64 = adjacency.iter().sum();
        (sum / n).min(2.0) // Proxy value
    }
}
