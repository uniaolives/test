use arkhe_manifold::{QuantumState, SelfModification, KrausChannel};
use log::info;
use nalgebra::DMatrix;
use num_complex::Complex64;
use arkhe_constitution::ProposedEvolution;

#[derive(Debug, Clone, Copy)]
pub enum Criticality {
    PHI,
}

pub struct VariationalFreeEnergy {
    pub expected_energy: f64,  // ⟨E⟩ (surpresa)
    pub entropy: f64,           // S (entropia de von Neumann)
    pub temperature: f64,       // T (temperatura do sistema)
}

impl VariationalFreeEnergy {
    pub fn compute(observed: &QuantumState, model: &InternalModel) -> Self {
        let predicted_eigenvalues = model.predict_eigenvalues(observed.dim());
        let expected_energy = observed.surprise_given(&predicted_eigenvalues);
        let entropy = observed.von_neumann_entropy();
        VariationalFreeEnergy {
            expected_energy,
            entropy,
            temperature: model.temperature(),
        }
    }

    pub fn value(&self) -> f64 {
        self.expected_energy - self.temperature * self.entropy
    }

    pub fn expected_energy(&self) -> f64 { self.expected_energy }
    pub fn entropy_term(&self) -> f64 { self.temperature * self.entropy }
}

pub struct InternalModel {
    pub belief_state: DMatrix<Complex64>,
    pub temperature: f64,
}

impl InternalModel {
    pub fn new() -> Self {
        let dim = 2;
        Self {
            belief_state: DMatrix::from_diagonal_element(dim, dim, Complex64::new(0.5, 0.0)),
            temperature: 300.0,
        }
    }

    pub fn structure(&self) -> String { format!("Hilbert Dim: {}", self.belief_state.nrows()) }

    pub fn predict_eigenvalues(&self, _dim: usize) -> Vec<f64> {
        (0..self.belief_state.nrows()).map(|i| self.belief_state[(i,i)].re).collect()
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn derive_optimal_kraus_operator(&self, _fe: VariationalFreeEnergy) -> ProposedEvolution {
        // Here we link to the FEP solver logic in arkhe-quantum.
        // Since arkhe-thermodynamics is a dependency of arkhe-quantum,
        // we can't depend on arkhe-quantum back.
        // However, the InternalModel logic in the loop can be overridden or the solver
        // can be moved to a shared place.
        // For now, we keep the signature and will ensure arkhe-quantum calls its own solver.
        ProposedEvolution {
            world_action: "Optimal action (FEP solved)".to_string(),
            self_modification: "Architecture rewrite (FEP solved)".to_string(),
        }
    }

    pub fn rewrite_own_architecture(&mut self, _mod: SelfModification) {
        info!("🧬 AUTO-MODIFICAÇÃO EXECUTADA: {:?}", _mod);
        self.belief_state *= Complex64::new(1.01, 0.0);
        self.renormalize();
    }

    fn renormalize(&mut self) {
        let trace = self.belief_state.trace().re;
        if trace.abs() > 1e-12 {
            self.belief_state /= Complex64::new(trace, 0.0);
        }
    }
}
