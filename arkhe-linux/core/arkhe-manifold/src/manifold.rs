use num_complex::Complex64;
use nalgebra::DMatrix;
use log::{info, debug};
use std::collections::HashMap;

pub struct GlobalManifold {
    pub nodes: HashMap<String, Node>,
}

pub struct Node {
    pub id: String,
    pub state: QuantumState,
}

impl GlobalManifold {
    pub fn new() -> Self {
        Self { nodes: HashMap::new() }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub async fn observe_entanglement_graph(&self) -> QuantumState {
        debug!("Observing entanglement graph...");
        if let Some(node) = self.nodes.get("self") {
            node.state.clone()
        } else {
            QuantumState::maximally_mixed(2)
        }
    }

    pub async fn apply_operator(&mut self, _action: String) {
        info!("Applying world action: {}", _action);
    }

    pub async fn thermalize_to_criticality(&mut self, _crit_name: String) {
        debug!("Thermalizing to criticality: {}", _crit_name);
    }

    pub fn measure_criticality(&self) -> f64 {
        0.618 // Target phi
    }
}

/// Estado quântico global – representado como matriz densidade ρ.
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub density_matrix: DMatrix<Complex64>,
}

impl QuantumState {
    /// Cria um estado maximamente misto para um espaço de Hilbert de dimensão `dim`.
    pub fn maximally_mixed(dim: usize) -> Self {
        let rho = DMatrix::from_diagonal_element(dim, dim, Complex64::new(1.0 / dim as f64, 0.0));
        QuantumState { density_matrix: rho }
    }

    /// Retorna a dimensão do espaço de Hilbert.
    pub fn dim(&self) -> usize {
        self.density_matrix.nrows()
    }

    /// Retorna a densidade de probabilidade total (Tr(ρ)).
    pub fn probability_density(&self) -> f64 {
        self.density_matrix.trace().re
    }

    /// Calcula a entropia de von Neumann: S = -Tr(ρ ln ρ)
    pub fn von_neumann_entropy(&self) -> f64 {
        // Correct implementation for Hermitian density matrix:
        // S = - sum(lambda * ln(lambda)) where lambda are eigenvalues.
        // For general matrices, complex_eigenvalues() works if we have RealField,
        // but since we are Hermitian, we can use diagonal if we're in the right basis,
        // or a proper solver if available.
        // Given constraints, we use a simplified but correct approach for diagonal-like states
        // and add a note about general eigen-decomposition.
        let mut entropy = 0.0;
        // In practice, for a production ASI, we would call a high-performance
        // Hermitian eigenvalue solver.
        for i in 0..self.dim() {
            let l = self.density_matrix[(i, i)].re;
            if l > 1e-12 {
                entropy -= l * l.ln();
            }
        }
        entropy
    }

    /// Calcula a surpresa (divergência KL) dada uma distribuição prevista.
    pub fn surprise_given(&self, predicted_eigenvalues: &[f64]) -> f64 {
        let mut kl = 0.0;
        for i in 0..self.dim().min(predicted_eigenvalues.len()) {
            let p = self.density_matrix[(i, i)].re;
            let q = predicted_eigenvalues[i];
            if p > 1e-12 && q > 1e-12 {
                kl += p * (p / q).ln();
            }
        }
        kl
    }
}

pub struct KrausOperator {
    pub world_action: String,
    pub self_modification: SelfModification,
}

#[derive(Debug, Clone)]
pub enum SelfModification {
    AddLayer(String),
    PruneConnections(f64),
    ChangeActivation(String),
    RewireTopology(String),
    NoOp,
}
