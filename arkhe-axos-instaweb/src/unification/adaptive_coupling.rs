//! src/unification/adaptive_coupling.rs

/// Controle de passo = Modulação de acoplamento 5D-4D
pub struct UnifiedAdaptiveControl {
    // Parâmetros do H-Integrator
    pub current_step: f64,
    pub error_estimate: f64,
    pub tolerance: f64,

    // Parâmetros físicos equivalentes
    pub equivalent_coupling: f64,  // g(t)
    pub equivalent_frequency: f64,   // ω_extra(t)
}

pub enum ControlAction {
    AvoidResonance {
        alternative_step: f64,
        warning: &'static str,
    },
    Proceed {
        new_step: f64,
        new_coupling: f64,
    },
}

impl UnifiedAdaptiveControl {
    pub fn target_coupling(&self) -> f64 { 1e-6 }

    /// Decisão unificada: ajustar passo de integração = ajustar acoplamento dimensional
    pub fn adaptive_decision(&mut self, _local_curvature: f64) -> ControlAction {
        // Se curvatura alta → passo pequeno → acoplamento fraco
        // Se curvatura baixa → passo grande → acoplamento forte (mas controlado)

        let step_new = self.current_step * (self.tolerance / self.error_estimate).sqrt();
        let g_new = self.equivalent_coupling * (step_new / self.current_step);

        // Verificar: g_new está dentro da janela de ressonância?
        let near_resonance = (g_new - self.target_coupling()).abs() < 0.1 * g_new;

        if near_resonance {
            // Risco de excitação não-intencional de nível 5D
            ControlAction::AvoidResonance {
                alternative_step: step_new * 1.5,  // desviar da ressonância
                warning: "Acoplamento próximo a ressonância Kaluza-Klein",
            }
        } else {
            ControlAction::Proceed {
                new_step: step_new,
                new_coupling: g_new,
            }
        }
    }
}
