use std::f64::consts::PI;
use num_complex::Complex64;
use nalgebra::{DMatrix, DVector};

pub struct PhaseBindingNetwork {
    pub oscillators: Vec<f64>,
    pub natural_frequencies: Vec<f64>,
    pub coupling_matrix: DMatrix<f64>,
    pub global_coupling: f64,
}

impl PhaseBindingNetwork {
    pub fn new(n: usize, global_coupling: f64) -> Self {
        Self {
            oscillators: vec![0.0; n],
            natural_frequencies: vec![1.0; n],
            coupling_matrix: DMatrix::from_element(n, n, 1.0),
            global_coupling,
        }
    }

    pub fn step(&mut self, dt: f64) {
        let n = self.oscillators.len();
        let mut new_phases = self.oscillators.clone();
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += self.coupling_matrix[(i, j)] * (self.oscillators[j] - self.oscillators[i]).sin();
                }
            }
            new_phases[i] = (self.oscillators[i] + self.natural_frequencies[i] * dt + self.global_coupling * sum * dt) % (2.0 * PI);
        }
        self.oscillators = new_phases;
    }
}
