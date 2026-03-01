use arkhe_quantum::{QuantumState, KrausOperator, Handover};
use std::collections::VecDeque;

pub struct QuantumAnomalyDetector {
    baseline_state: QuantumState,
    state_history: VecDeque<QuantumState>,
    entropy_threshold: f64,
    fidelity_threshold: f64,
}

impl QuantumAnomalyDetector {
    pub fn new(dim: usize) -> Self {
        Self {
            baseline_state: QuantumState::new(dim),
            state_history: VecDeque::with_capacity(1000),
            entropy_threshold: 0.1,
            fidelity_threshold: 0.95,
        }
    }

    pub fn observe_handover(&mut self, h: &Handover) -> Option<SecurityAlert> {
        let op = KrausOperator::from_handover(h, self.baseline_state.dim());
        let mut current = self.baseline_state.clone();
        current.evolve(&op);

        let entropy = current.von_neumann_entropy();
        let fidelity = self.baseline_state.fidelity(&current);

        if (entropy - self.baseline_state.von_neumann_entropy()).abs() > self.entropy_threshold {
            return Some(SecurityAlert::EntropyDeviation(entropy));
        }
        if fidelity < self.fidelity_threshold {
            return Some(SecurityAlert::LowFidelity(fidelity));
        }

        self.state_history.push_back(current);
        if self.state_history.len() == self.state_history.capacity() {
            self.recompute_baseline();
        }

        None
    }

    fn recompute_baseline(&mut self) {
        if let Some(latest) = self.state_history.back() {
            self.baseline_state = latest.clone();
        }
    }
}

#[derive(Debug, Clone)]
pub enum SecurityAlert {
    EntropyDeviation(f64),
    LowFidelity(f64),
    PossibleIntrusion(String),
}
