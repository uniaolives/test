//! quantum_simulation.rs
//! Stubs for quantum simulation primitives used by MetaConsciousness

use nalgebra::{DVector, Complex};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantumState {
    pub amplitudes: DVector<Complex<f64>>,
}

impl QuantumState {
    pub fn new(size: usize) -> Self {
        let mut amplitudes = DVector::from_element(size, Complex::new(0.0, 0.0));
        if size > 0 {
            amplitudes[0] = Complex::new(1.0, 0.0);
        }
        Self { amplitudes }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumGate {
    pub matrix: nalgebra::DMatrix<Complex<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceModel {
    pub rate: f64,
}
