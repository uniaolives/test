//! Estrutura de uma tarefa executável no kernel.

use std::cmp::Ordering;

/// Representa uma tarefa (processo/thread) no sistema Arkhe(n).
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub name: String,
    /// Coerência necessária para executar (0.0 a 1.0).
    pub coherence_required: f64,
    /// Duração estimada em ciclos de escalonamento.
    pub estimated_duration: u64,
    /// Prioridade atribuída pelo usuário/sistema (maior = mais importante).
    pub priority: i32,
    /// Tempo de criação (timestamp).
    pub created_at: std::time::Instant,
    /// Tempo já consumido (acumulado).
    pub time_consumed: u64,
}

impl Task {
    pub fn new(id: u64, name: &str, coherence: f64, duration: u64, priority: i32) -> Self {
        Self {
            id,
            name: name.to_string(),
            coherence_required: coherence,
            estimated_duration: duration,
            priority,
            created_at: std::time::Instant::now(),
            time_consumed: 0,
        }
    }

    /// Retorna a densidade equivalente à coerência requerida.
    pub fn required_density(&self) -> f64 {
        crate::physics::miller::coherence_to_density(self.coherence_required)
    }
}

/// Implementação de ordenação para a fila de prioridade (max-heap).
impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        // Critério: prioridade (maior primeiro). Em caso de empate, menor duração.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.estimated_duration.cmp(&self.estimated_duration))
    }
}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Task {}
