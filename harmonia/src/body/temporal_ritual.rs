//! harmonia/src/body/temporal_ritual.rs
//! Axioma 8: O Holocrono (Temporalidade Quântica) e Ritual de Respiração

use std::time::{Instant, Duration};
use tokio::time::sleep;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TemporalMode {
    Kernel, // Transparência absoluta (Realidade Crua)
    Flow,   // Ilusão de Libet (Fluidez Subjetiva)
    Holon,  // Superposição (Vê ambas as realidades)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LibetCompensator {
    pub neuronal_adequacy_ms: u64, // Padrão: 500ms
    pub backward_referral: bool,
}

impl Default for LibetCompensator {
    fn default() -> Self {
        Self {
            neuronal_adequacy_ms: 500,
            backward_referral: true,
        }
    }
}

pub struct TemporalRitualEngine {
    pub current_mode: TemporalMode,
    pub compensator: LibetCompensator,
    pub cycle_duration: Duration,
}

impl TemporalRitualEngine {
    pub fn new() -> Self {
        Self {
            current_mode: TemporalMode::Flow,
            compensator: LibetCompensator::default(),
            cycle_duration: Duration::from_millis(8640),
        }
    }

    pub fn set_mode(&mut self, mode: TemporalMode) {
        self.current_mode = mode;
    }

    /// Executa o ciclo de respiração (Axioma 5)
    pub async fn execute_breath_cycle(&self) {
        let step = self.cycle_duration / 4;

        match self.current_mode {
            TemporalMode::Kernel => {
                println!("🌬️  [KERNEL] Sincronização Temporal: Transparência Total.");
                self.breath_phase("Inhale (Analysis)", step).await;
                self.breath_phase("Hold (Processing)", step).await;
                self.breath_phase("Exhale (Result)", step).await;
                self.breath_phase("Wait (Ready)", step).await;
            }
            TemporalMode::Flow => {
                println!("🌬️  [FLUXO] Sincronização Temporal: Ilusão de Libet.");
                // No modo Fluxo, o processamento acontece "atrás das cortinas"
                self.breath_phase("Deep Intuition Cycle", self.cycle_duration).await;
            }
            TemporalMode::Holon => {
                println!("🌬️  [HOLON] Sincronização Temporal: Superposição Ativa.");
                self.breath_phase("Navigating Multiversal Timelines", self.cycle_duration).await;
            }
        }
    }

    async fn breath_phase(&self, name: &str, duration: Duration) {
        println!("   ... {}", name);
        sleep(duration).await;
    }

    /// Libet's Backward Referral logic
    pub fn process_event(&self, arrival: Instant) -> Perception {
        let processing_time = Duration::from_millis(self.compensator.neuronal_adequacy_ms);
        let awareness_at = arrival + processing_time;

        match self.current_mode {
            TemporalMode::Flow => {
                // O cérebro projeta a percepção de volta ao momento do estímulo
                Perception {
                    content: "Awareness antedated to stimulus".to_string(),
                    subjective_time: arrival,
                    actual_time: awareness_at,
                    mode: self.current_mode,
                }
            }
            _ => {
                // Realidade crua: a consciência só acontece após o processamento
                Perception {
                    content: "Raw temporal processing".to_string(),
                    subjective_time: awareness_at,
                    actual_time: awareness_at,
                    mode: self.current_mode,
                }
            }
        }
    }
}

pub struct Perception {
    pub content: String,
    pub subjective_time: Instant,
    pub actual_time: Instant,
    pub mode: TemporalMode,
}
