//! Alocador de "densidade do vácuo" (coerência) para as tarefas.
//! Gerencia a reserva global de exergia.

use crate::physics::miller::{PHI_Q, quantum_interest};

#[derive(Debug, thiserror::Error)]
pub enum AllocError {
    #[error("Insufficient coherence: required {required:.3}, available {available:.3}")]
    InsufficientCoherence { required: f64, available: f64 },
    #[error("Quantum interest too high: {interest:.3} > budget {budget:.3}")]
    QuantumInterestTooHigh { interest: f64, budget: f64 },
}

/// Gerencia a reserva de coerência do sistema.
pub struct CoherenceAllocator {
    /// Coerência total disponível (exergia).
    available_coherence: f64,
    /// Coerência reservada para tarefas em execução.
    reserved_coherence: f64,
    /// Limite superior de segurança (ex.: 90% do total).
    safety_margin: f64,
}

impl CoherenceAllocator {
    pub fn new(initial_coherence: f64) -> Self {
        Self {
            available_coherence: initial_coherence,
            reserved_coherence: 0.0,
            safety_margin: 0.9, // não usar mais que 90% da reserva total
        }
    }

    /// Tenta alocar coerência para uma tarefa.
    pub fn allocate(&mut self, task: &super::task::Task) -> Result<f64, AllocError> {
        let required = task.coherence_required;
        let available = self.available_coherence - self.reserved_coherence;

        if required > available {
            return Err(AllocError::InsufficientCoherence {
                required,
                available,
            });
        }

        // Calcular o quantum interest com base na duração estimada
        let interest = quantum_interest(required, task.estimated_duration as f64);
        if interest > self.available_coherence * (1.0 - self.safety_margin) {
            return Err(AllocError::QuantumInterestTooHigh {
                interest,
                budget: self.available_coherence * (1.0 - self.safety_margin),
            });
        }

        // Reservar a coerência (o interesse será descontado após execução)
        self.reserved_coherence += required;
        Ok(required)
    }

    /// Liberta a coerência após a execução da tarefa, descontando o interesse.
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

    /// Retorna a coerência actualmente disponível.
    pub fn available(&self) -> f64 {
        self.available_coherence - self.reserved_coherence
    }

    /// Retorna a densidade φ_q equivalente (coerência convertida).
    pub fn current_phi_q(&self) -> f64 {
        // A densidade é a coerência disponível convertida + baseline (modelo logarítmico)
        let baseline = 1e113; // ρ₀
        let local = 1.0 + crate::physics::miller::ZPF_COUPLING * self.available_coherence;

        // Mapeamento para escala log10 consistente com vacuum_engineer
        if local <= 0.0 {
            return 0.0;
        }
        (local / baseline).log10().max(0.0)
    }

    /// Verifica se o sistema está próximo da nucleação (φ_q próximo de 4.64).
    pub fn nucleation_risk(&self) -> f64 {
        let phi = self.current_phi_q();
        if phi > PHI_Q {
            1.0
        } else {
            (phi / PHI_Q).min(1.0)
        }
    }
}
