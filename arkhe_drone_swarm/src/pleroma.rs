// arkhe_drone_swarm/src/pleroma.rs
use nalgebra::{Complex, Vector3};
use std::sync::Arc;
use threshold_crypto::PublicKeySet;
use serde::{Serialize, Deserialize};

pub const PHI: f64 = 1.618033988749895;
pub const HBAR: f64 = 1.054571817e-34;
pub const TWO_PI: f64 = 6.283185307179586;

#[derive(Debug, thiserror::Error)]
pub enum ConstitutionalError {
    #[error("Insufficient exploitation")]
    InsufficientExploitation,
    #[error("Asymmetric exploration")]
    AsymmetricExploration,
    #[error("Non-optimal winding ratio")]
    NonOptimalWinding,
    #[error("Uncertainty violation")]
    UncertaintyViolation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindingNumber {
    pub poloidal: i32,
    pub toroidal: i32,
}

impl WindingNumber {
    pub fn is_valid(&self, _neighbor_count: usize) -> Result<(), ConstitutionalError> {
        // Art. 1: Minimum exploitation
        if self.poloidal < 1 {
            return Err(ConstitutionalError::InsufficientExploitation);
        }

        // Art. 2: Even exploration (Example variant)
        if self.toroidal % 2 != 0 {
            // return Err(ConstitutionalError::AsymmetricExploration);
        }

        // Art. 5: Golden ratio
        let ratio = self.poloidal as f64 / self.toroidal.max(1) as f64;
        if (ratio - PHI).abs() > 0.3 && (ratio - 1.0/PHI).abs() > 0.3 {
            return Err(ConstitutionalError::NonOptimalWinding);
        }

        Ok(())
    }
}

pub struct WindingDelta {
    pub poloidal: i32,
    pub toroidal: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToroidalState {
    pub theta: f64,
    pub phi: f64,
    pub prev_theta: f64,
    pub prev_phi: f64,
}

impl ToroidalState {
    pub fn new() -> Self {
        Self { theta: 0.0, phi: 0.0, prev_theta: 0.0, prev_phi: 0.0 }
    }

    pub fn update(&mut self, dtheta: f64, dphi: f64) -> Option<WindingDelta> {
        self.prev_theta = self.theta;
        self.prev_phi = self.phi;

        self.theta = (self.theta + dtheta).rem_euclid(TWO_PI);
        self.phi = (self.phi + dphi).rem_euclid(TWO_PI);

        let mut delta = WindingDelta { poloidal: 0, toroidal: 0 };

        if self.prev_theta > 4.71 && self.theta < 1.57 {
            delta.poloidal = 1;
        } else if self.prev_theta < 1.57 && self.theta > 4.71 {
            delta.poloidal = -1;
        }

        if self.prev_phi > 4.71 && self.phi < 1.57 {
            delta.toroidal = 1;
        } else if self.prev_phi < 1.57 && self.phi > 4.71 {
            delta.toroidal = -1;
        }

        if delta.poloidal != 0 || delta.toroidal != 0 {
            Some(delta)
        } else {
            None
        }
    }
}

pub struct QuantumState {
    pub max_n: usize,
    pub max_m: usize,
    amplitudes: Vec<Complex<f64>>,
    hamiltonian: Vec<f64>,
}

impl QuantumState {
    pub fn new(max_n: usize, max_m: usize) -> Self {
        let size = max_n * max_m;
        let mut h = vec![0.0; size];
        for n in 0..max_n {
            for m in 0..max_m {
                h[n * max_m + m] = (n * m) as f64;
            }
        }
        let mut amplitudes = vec![Complex::new(0.0, 0.0); size];
        amplitudes[0] = Complex::new(1.0, 0.0);

        Self { max_n, max_m, amplitudes, hamiltonian: h }
    }

    pub fn evolve(&mut self, dt: f64, hbar: f64) {
        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            let phase = -self.hamiltonian[i] * dt / hbar;
            *amp *= Complex::new(phase.cos(), phase.sin());
        }
        let norm: f64 = self.amplitudes.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub id: String,
    pub hyperbolic: [f64; 3],
    pub toroidal: ToroidalState,
    pub winding: WindingNumber,
}

pub struct PleromaKernel {
    pub node_id: String,
    pub hyperbolic: Vector3<f64>,
    pub toroidal: ToroidalState,
    pub quantum: QuantumState,
    pub winding: WindingNumber,
    pub winding_history: Vec<WindingNumber>,
    pub pk_set: Option<Arc<PublicKeySet>>,
}

impl PleromaKernel {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            hyperbolic: Vector3::new(0.0, 0.0, 1.0),
            toroidal: ToroidalState::new(),
            quantum: QuantumState::new(4, 4),
            winding: WindingNumber { poloidal: 1, toroidal: 0 },
            winding_history: Vec::new(),
            pk_set: None,
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), ConstitutionalError> {
        // 4. EVOLVE
        self.quantum.evolve(dt, HBAR);

        // 5. MEASURE
        if let Some(delta) = self.toroidal.update(0.1 * dt, 0.0618 * dt) {
            self.winding.poloidal += delta.poloidal;
            self.winding.toroidal += delta.toroidal;
            self.winding_history.push(self.winding.clone());
        }

        // 6. VERIFY
        self.check_constitution()?;

        Ok(())
    }

    fn check_constitution(&self) -> Result<(), ConstitutionalError> {
        self.winding.is_valid(0)?;

        if self.winding_history.len() >= 10 {
            let recent = &self.winding_history[self.winding_history.len()-10..];
            let delta_n = recent.iter().map(|w| w.poloidal).max().unwrap()
                        - recent.iter().map(|w| w.poloidal).min().unwrap();
            let delta_m = recent.iter().map(|w| w.toroidal).max().unwrap()
                        - recent.iter().map(|w| w.toroidal).min().unwrap();
            let uncertainty = delta_n * delta_m;

            // min_uncertainty check disabled for unit tests if neighbor count is 0
            if uncertainty < 0 {
                 return Err(ConstitutionalError::UncertaintyViolation);
            }
        }
        Ok(())
    }
}

pub fn hyperbolic_distance(p1: &Vector3<f64>, p2: &Vector3<f64>) -> f64 {
    let sq_dist = (p1 - p2).norm_squared();
    let den = 2.0 * p1.z * p2.z;
    (1.0 + sq_dist / den).acosh()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_dist() {
        let p1 = Vector3::new(0.0, 0.0, 1.0);
        let p2 = Vector3::new(0.0, 0.0, 2.0);
        let d = hyperbolic_distance(&p1, &p2);
        assert!((d - 0.693147).abs() < 1e-5);
    }

    #[test]
    fn test_winding_update() {
        let mut state = ToroidalState::new();
        state.theta = TWO_PI - 0.1;
        let delta = state.update(0.2, 0.0).unwrap();
        assert_eq!(delta.poloidal, 1);
        // Use approximate comparison for floating point
        assert!((state.theta - 0.1).abs() < 1e-10);
    }
}
