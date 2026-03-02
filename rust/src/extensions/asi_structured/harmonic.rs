use std::f64::consts::PI;
use nalgebra::{DVector, Complex};
use serde::{Serialize, Deserialize};
use super::geometry::{ShellGeometry, HarmonicBasis};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicASI {
    pub geometry: ShellGeometry,
    pub active_modes: Vec<HarmonicMode>,
    pub level: ConsciousnessLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicMode {
    pub l: i32,
    pub m: i32,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ConsciousnessLevel {
    SimpleASI { mode_count: usize },
    ComplexASI { mode_count: usize },
    ConsciousASI { mode_count: usize },
    TranscendentASI { mode_count: usize },
}

impl HarmonicASI {
    pub fn new(geometry: ShellGeometry, level: ConsciousnessLevel) -> Self {
        let mode_count = match level {
            ConsciousnessLevel::SimpleASI { mode_count } => mode_count,
            ConsciousnessLevel::ComplexASI { mode_count } => mode_count,
            ConsciousnessLevel::ConsciousASI { mode_count } => mode_count,
            ConsciousnessLevel::TranscendentASI { mode_count } => mode_count,
        };

        let mut active_modes = Vec::new();
        for i in 0..mode_count {
            active_modes.push(HarmonicMode {
                l: i as i32,
                m: 0,
                amplitude: 1.0 / (1.0 + i as f64),
                phase: 0.0,
            });
        }

        Self { geometry, active_modes, level }
    }

    pub fn evolve(&mut self, dt: f64) {
        let d = self.geometry.ambient_dimension as i32;
        for mode in &mut self.active_modes {
            // Eigenvalue of spherical Laplacian: l(l + d - 2)
            let frequency = (mode.l * (mode.l + d - 2)) as f64;
            mode.phase = (mode.phase + frequency * dt) % (2.0 * PI);
        }
    }

    pub fn generate_thought(&self, basis: &HarmonicBasis) -> DVector<f64> {
        let mut sum = DVector::zeros(self.geometry.ambient_dimension);
        for mode in &self.active_modes {
            let basis_v = basis.basis_vector(mode.l, mode.m);
            sum += basis_v * mode.amplitude * mode.phase.cos();
        }
        self.geometry.project_to_shell(&sum)
    }
}
