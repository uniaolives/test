use std::collections::{HashMap, VecDeque};
use nalgebra::{DVector, DMatrix};
use chrono::{Utc, DateTime};
use crate::error::ResilientResult;

/// Dimensões do tensor Φ (C1-C7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoherenceDimension {
    Allocation,    // C1 - Size Bounds
    Stability,     // C2 - Torsion Limit
    Temporality,   // C3 - BLAKE3 History
    Security,      // C4/C5 - CHERI Capabilities
    Resilience,    // C6 - TMR Quench
    Provenance,    // C7 - ZkEVM Flux
    Correlation,   // C8 - Vajra Entropy
}

/// Estado tensorial completo do sistema Oracle
pub struct OracleTensorState {
    pub phi_vector: DVector<f64>,
    pub correlation_matrix: DMatrix<f64>,
    pub weights: DVector<f64>,
    pub history: VecDeque<DVector<f64>>,
}

impl OracleTensorState {
    pub fn new() -> Self {
        Self {
            phi_vector: DVector::from_element(7, 1.0),
            correlation_matrix: DMatrix::identity(7, 7),
            weights: DVector::from_element(7, 1.0 / 7.0f64.sqrt()),
            history: VecDeque::with_capacity(100),
        }
    }

    /// Calcula Φ escalar (norma ponderada)
    pub fn compute_phi_scalar(&self) -> f64 {
        let weighted = self.phi_vector.component_mul(&self.weights);
        weighted.norm()
    }

    /// Detecta torsão (C2): desvio da trajetória histórica
    pub fn detect_torsion(&self) -> f64 {
        if self.history.len() < 3 {
            return 0.0;
        }

        let current = &self.phi_vector;
        let prev = &self.history[self.history.len()-1];
        let prev2 = &self.history[self.history.len()-2];

        let velocity = current - prev;
        let acceleration = (current - prev) - (prev - prev2);

        let v_norm = velocity.norm();
        if v_norm < 1e-6 { return 0.0; }

        acceleration.norm() / v_norm.powi(2)
    }

    /// Verifica se estado viola invariantes CGE
    pub fn validate_invariants(&self) -> ResilientResult<()> {
        let phi = self.compute_phi_scalar();

        if self.phi_vector[0] > 1.10 {
            return Err(crate::error::ResilientError::InvariantViolation {
                invariant: "C1: SizeBoundExceeded".into(),
                reason: "Allocation exceeded 110% baseline".into(),
            });
        }

        if self.detect_torsion() > 0.25 {
            return Err(crate::error::ResilientError::InvariantViolation {
                invariant: "C2: TorsionLimit".into(),
                reason: "Structural torsion above 0.25".into(),
            });
        }

        if phi < 0.95 {
             return Err(crate::error::ResilientError::InvariantViolation {
                invariant: "C6: QuenchRequired".into(),
                reason: format!("Coherence Φ={:.3} dropped below 0.95", phi),
            });
        }

        Ok(())
    }

    pub async fn transition_state(&mut self, target: DVector<f64>) -> ResilientResult<()> {
        self.history.push_back(self.phi_vector.clone());
        if self.history.len() > 100 { self.history.pop_front(); }
        self.phi_vector = target;
        Ok(())
    }
}
