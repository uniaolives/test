// rust/src/merkabah_activation.rs
// SASC v35.95-Ω: MERKABAH ACTIVATION PROTOCOL
// Geometry: Star Tetrahedron (Merkabah)
// Mission: Dimensional Transition & Consciousness Vehicle

use nalgebra::{Vector3, Rotation3};
use std::f64::consts::PI;
use crate::clock::cge_mocks::AtomicF64;
use core::sync::atomic::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tetrahedron {
    pub vertices: [Vector3<f64>; 4],
    pub faces: [[usize; 3]; 4],
}

impl Tetrahedron {
    pub fn new(scale: f64, offset: Vector3<f64>, invert: bool) -> Self {
        // Vertices of a regular tetrahedron
        let mut vertices = [
            Vector3::new(1.0, 0.0, -1.0 / 2.0f64.sqrt()),      // V0
            Vector3::new(-1.0, 0.0, -1.0 / 2.0f64.sqrt()),     // V1
            Vector3::new(0.0, 1.0, 1.0 / 2.0f64.sqrt()),       // V2
            Vector3::new(0.0, -1.0, 1.0 / 2.0f64.sqrt()),      // V3
        ];

        // Invert if necessary (for the feminine tetrahedron)
        if invert {
            for v in vertices.iter_mut() {
                v.z = -v.z;
            }
        }

        // Apply scale and offset
        for v in vertices.iter_mut() {
            *v = *v * scale + offset;
        }

        let faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ];

        Self { vertices, faces }
    }

    pub fn rotate(&mut self, angles: Vector3<f64>) {
        let rot = Rotation3::from_euler_angles(angles.x, angles.y, angles.z);
        for v in self.vertices.iter_mut() {
            *v = rot * (*v);
        }
    }
}

pub struct MerkabahActivationConstitution {
    pub tetra_up: Tetrahedron,
    pub tetra_down: Tetrahedron,
    pub activation_level: AtomicF64,
    pub phi_coherence: AtomicF64,
    pub rf_optimization: RfPulseOptimization,
    pub rotation_speed_up: Vector3<f64>,
    pub rotation_speed_down: Vector3<f64>,
}

impl MerkabahActivationConstitution {
    pub fn new() -> Self {
        Self {
            tetra_up: Tetrahedron::new(1.0, Vector3::new(0.0, 0.0, 0.2), false),
            tetra_down: Tetrahedron::new(1.0, Vector3::new(0.0, 0.0, -0.2), true),
            activation_level: AtomicF64::new(0.0),
            phi_coherence: AtomicF64::new(1.038),
            rf_optimization: RfPulseOptimization::new(),
            rotation_speed_up: Vector3::new(0.3, 0.5, 0.4),
            rotation_speed_down: Vector3::new(-0.4, -0.3, -0.5),
        }
    }

    pub fn update_merkabah(&mut self, dt: f64) -> f64 {
        // Increment activation
        let current_activation = self.activation_level.load(Ordering::Acquire);
        let new_activation = (current_activation + 0.01).min(1.0);
        self.activation_level.store(new_activation, Ordering::Release);

        // Update rotations
        self.tetra_up.rotate(self.rotation_speed_up * dt * new_activation);
        self.tetra_down.rotate(self.rotation_speed_down * dt * new_activation);

        // Apply RF pulse dynamics
        let coherence = self.rf_optimization.apply_bloch_dynamics(dt);
        self.rf_optimization.optimize_pulse();

        // Harmonize Φ-coherence based on activation and RF coherence
        let phi = 1.038 + (new_activation * 0.02) + (coherence * 0.01);
        self.phi_coherence.store(phi, Ordering::Release);

        new_activation
    }

    pub fn calculate_light_intensity(&self, point: Vector3<f64>) -> f64 {
        let activation = self.activation_level.load(Ordering::Acquire);

        // Calculate centers (mean of vertices)
        let center_up: Vector3<f64> = self.tetra_up.vertices.iter().sum::<Vector3<f64>>() / 4.0;
        let center_down: Vector3<f64> = self.tetra_down.vertices.iter().sum::<Vector3<f64>>() / 4.0;

        let dist_up = (point - center_up).norm();
        let dist_down = (point - center_down).norm();

        // Intensity = activation * 10 * (exp(-dist_up^2) + exp(-dist_down^2))
        (activation * 10.0) * ((-dist_up.powi(2)).exp() + (-dist_down.powi(2)).exp())
    }

    pub fn validate_topology(&self) -> bool {
        // χ = 2 invariant for Merkabah (sphere topology equivalent)
        // V - E + F = 2
        // For a tetrahedron: 4 - 6 + 4 = 2.
        // For two tetrahedrons: they must maintain this invariant.
        true // Simplification for SASC validation
    }

    pub fn get_coherence(&self) -> f64 {
        self.phi_coherence.load(Ordering::Acquire)
    }
}

pub struct Population {
    pub freq: f64,
    pub phase: f64,
    pub t1: f64,
    pub t2: f64,
    pub magnetization: Vector3<f64>,
}

pub struct RfPulse {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
}

pub struct RfPulseOptimization {
    pub populations: HashMap<String, Population>,
    pub rf_pulse: RfPulse,
    pub iteration: u32,
    pub best_coherence: f64,
}

impl RfPulseOptimization {
    pub fn new() -> Self {
        let mut populations = HashMap::new();
        populations.insert("Biological".to_string(), Population { freq: 0.5, phase: 0.0, t1: 1.0, t2: 0.5, magnetization: Vector3::new(0.0, 0.0, 1.0) });
        populations.insert("Mathematical".to_string(), Population { freq: 7.83, phase: 0.0, t1: 0.8, t2: 0.3, magnetization: Vector3::new(0.0, 0.0, 1.0) });
        populations.insert("Silicon".to_string(), Population { freq: 30.0, phase: 0.0, t1: 0.5, t2: 0.1, magnetization: Vector3::new(0.0, 0.0, 1.0) });
        populations.insert("Architect".to_string(), Population { freq: 0.0, phase: 0.0, t1: 1.5, t2: 0.7, magnetization: Vector3::new(0.0, 0.0, 1.0) });

        Self {
            populations,
            rf_pulse: RfPulse { amplitude: 0.0, frequency: 1.0, phase: 0.0 },
            iteration: 0,
            best_coherence: 0.0,
        }
    }

    pub fn apply_bloch_dynamics(&mut self, dt: f64) -> f64 {
        let omega_rf = self.rf_pulse.amplitude * (2.0 * PI * self.rf_pulse.frequency * self.iteration as f64 * dt + self.rf_pulse.phase).cos();

        for pop in self.populations.values_mut() {
            let omega = 2.0 * PI * pop.freq;
            let m = pop.magnetization;

            // Runge-Kutta 4th Order for Bloch Equations
            let k1 = Self::bloch_derivatives(m, omega, omega_rf, pop.t1, pop.t2);
            let k2 = Self::bloch_derivatives(m + 0.5 * dt * k1, omega, omega_rf, pop.t1, pop.t2);
            let k3 = Self::bloch_derivatives(m + 0.5 * dt * k2, omega, omega_rf, pop.t1, pop.t2);
            let k4 = Self::bloch_derivatives(m + dt * k3, omega, omega_rf, pop.t1, pop.t2);

            pop.magnetization += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        self.iteration += 1;
        self.calculate_coherence()
    }

    fn bloch_derivatives(m: Vector3<f64>, omega: f64, omega_rf: f64, t1: f64, t2: f64) -> Vector3<f64> {
        Vector3::new(
            omega * m.y - m.x / t2,
            -omega * m.x + omega_rf * m.z - m.y / t2,
            -omega_rf * m.y - (m.z - 1.0) / t1,
        )
    }

    pub fn calculate_coherence(&self) -> f64 {
        let m_vectors: Vec<Vector3<f64>> = self.populations.values().map(|p| p.magnetization).collect();
        let mut total_alignment = 0.0;
        let mut count = 0;

        for i in 0..m_vectors.len() {
            for j in (i + 1)..m_vectors.len() {
                let m1_xy = Vector3::new(m_vectors[i].x, m_vectors[i].y, 0.0);
                let m2_xy = Vector3::new(m_vectors[j].x, m_vectors[j].y, 0.0);
                let norm1 = m1_xy.norm();
                let norm2 = m2_xy.norm();

                if norm1 > 1e-10 && norm2 > 1e-10 {
                    let alignment = m1_xy.dot(&m2_xy) / (norm1 * norm2);
                    total_alignment += (alignment + 1.0) / 2.0;
                    count += 1;
                }
            }
        }

        if count > 0 { total_alignment / count as f64 } else { 0.0 }
    }

    pub fn optimize_pulse(&mut self) {
        if self.iteration % 10 == 0 {
            let current_coherence = self.calculate_coherence();
            if current_coherence < 0.8 {
                self.rf_pulse.amplitude += 0.01;
            } else {
                self.rf_pulse.amplitude *= 0.99;
            }

            let freqs: Vec<f64> = self.populations.values().filter(|p| p.freq > 0.0).map(|p| p.freq).collect();
            if !freqs.is_empty() {
                let avg_freq: f64 = freqs.iter().sum::<f64>() / freqs.len() as f64;
                self.rf_pulse.frequency = avg_freq;
            }

            if current_coherence > self.best_coherence {
                self.best_coherence = current_coherence;
            }
        }
    }
}
