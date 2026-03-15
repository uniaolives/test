use ndarray::{Array1, Array2};
use num_complex::Complex;
use std::f64::consts::PI;

pub struct PhaseMemory {
    pub n_modes: usize,
    pub yang_baxter: Array2<Complex<f64>>,
}

impl PhaseMemory {
    pub fn new(n_modes: usize) -> Self {
        Self {
            n_modes,
            yang_baxter: Self::build_yang_baxter_matrix(n_modes),
        }
    }

    fn build_yang_baxter_matrix(n: usize) -> Array2<Complex<f64>> {
        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let phase = (i * j) as f64 * PI / (2.0 * n as f64);
                matrix[[i, j]] = Complex::from_polar(1.0, phase);
            }
        }
        matrix
    }

    pub fn solve_white_dispersion(
        &self,
        initial_state: &Array1<f64>,
        diffusion_coeff: f64,
        dt: f64,
        steps: usize,
    ) -> Array1<f64> {
        let n = initial_state.len();
        let mut state = initial_state.clone();
        for _ in 0..steps {
            let mut laplacian = Array1::zeros(n);
            for i in 1..n-1 {
                laplacian[i] = state[i-1] - 2.0 * state[i] + state[i+1];
            }
            state = &state + &(&laplacian * diffusion_coeff) * dt;
        }
        state
    }

    pub fn compute_lambda_2(&self, adjacency: &Array2<f64>) -> f64 {
        // Simplified eigenvalue extraction proxy
        let n = adjacency.nrows() as f64;
        let sum: f64 = adjacency.iter().sum();
        (sum / n).min(1.0)
    }
}
