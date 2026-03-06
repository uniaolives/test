use crate::lib::miller::{PHI_Q, quantum_interest};

#[derive(Debug, thiserror::Error)]
pub enum AllocError {
    #[error("Insufficient coherence: required {required:.3}, available {available:.3}")]
    InsufficientCoherence { required: f64, available: f64 },
    #[error("Quantum interest too high: {interest:.3} > budget {budget:.3}")]
    QuantumInterestTooHigh { interest: f64, budget: f64 },
}

pub struct CoherenceAllocator {
    available_coherence: f64,
    reserved_coherence: f64,
    safety_margin: f64,
}

impl CoherenceAllocator {
    pub fn new(initial_coherence: f64) -> Self {
        Self {
            available_coherence: initial_coherence,
            reserved_coherence: 0.0,
            safety_margin: 0.9,
        }
    }

    pub fn allocate(&mut self, task: &super::task::Task) -> Result<f64, AllocError> {
        let required = task.coherence_required;
        let available = self.available_coherence - self.reserved_coherence;

        if required > available {
            return Err(AllocError::InsufficientCoherence {
                required,
                available,
            });
        }

        let interest = quantum_interest(required, task.estimated_duration as f64);
        if interest > self.available_coherence * (1.0 - self.safety_margin) {
            return Err(AllocError::QuantumInterestTooHigh {
                interest,
                budget: self.available_coherence * (1.0 - self.safety_margin),
            });
        }

        self.reserved_coherence += required;
        Ok(required)
    }

    pub fn free(&mut self, task: &super::task::Task) {
        let interest = quantum_interest(task.coherence_required, task.estimated_duration as f64);
        let consumed = task.coherence_required + interest;

        if self.reserved_coherence >= task.coherence_required {
            self.reserved_coherence -= task.coherence_required;
        } else {
            self.reserved_coherence = 0.0;
        }

        if self.available_coherence >= consumed {
            self.available_coherence -= consumed;
        } else {
            self.available_coherence = 0.0;
        }
    }

    pub fn available(&self) -> f64 {
        self.available_coherence - self.reserved_coherence
    }

    pub fn current_phi_q(&self) -> f64 {
        1.0 + crate::lib::miller::ZPF_COUPLING * self.available_coherence
    }

    pub fn nucleation_risk(&self) -> f64 {
        let phi = self.current_phi_q();
        if phi > PHI_Q { 1.0 } else { (phi / PHI_Q).min(1.0) }
    }
}
