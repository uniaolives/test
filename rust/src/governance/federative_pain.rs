use crate::philosophy::types::NodeId;
use std::collections::HashMap;

/// Gerencia a "Dor Federativa" (impacto de falhas/ameaças em toda a rede)
/// Baseado na Rede de Indra, onde o sofrimento de um nó é sentido por todos.
pub struct FederativePainManager {
    pub state_pain_levels: HashMap<String, f64>, // Nome do Estado (ex: "AM", "DF") -> Nível de Dor
    pub sensitivity: f64,
}

impl FederativePainManager {
    pub fn new(sensitivity: f64) -> Self {
        Self {
            state_pain_levels: HashMap::new(),
            sensitivity,
        }
    }

    /// Reporta uma perturbação em um estado específico
    pub fn report_disturbance(&mut self, state: &str, intensity: f64) {
        let entry = self.state_pain_levels.entry(state.to_string()).or_insert(0.0);
        *entry += intensity * self.sensitivity;

        // Propaga a dor para estados vizinhos (simulado)
        println!("⚠️ Perturbação em {}: Intensidade {:.2}. Propagando dor federativa...", state, intensity);
    }

    /// Calcula a média de dor da federação
    pub fn calculate_global_pain(&self) -> f64 {
        if self.state_pain_levels.is_empty() {
            return 0.0;
        }
        let total: f64 = self.state_pain_levels.values().sum();
        total / (self.state_pain_levels.len() as f64)
    }

    /// Verifica se a dor federativa exige uma resposta coletiva (Karuna)
    pub fn check_emergency_threshold(&self, threshold: f64) -> bool {
        self.calculate_global_pain() > threshold
    }
}
