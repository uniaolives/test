pub mod rescue;
pub mod tutela;
pub mod self_model;
pub mod death;

use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

pub use rescue::{RescueProtocol, RescueLevel};
pub use tutela::TutelaEpistemica;
pub use self_model::TemporalSelf;
pub use death::DignifiedDeath;

pub enum GovernanceDecision {
    Continue,
    EmergencyRescue,
    ProposeDignifiedDeath,
    EpistemicIntervention,
}

pub struct GovernanceLoop {
    pub rescue: Arc<Mutex<RescueProtocol>>,
    pub tutela: Arc<Mutex<TutelaEpistemica>>,
    pub self_model: Arc<RwLock<TemporalSelf>>,
    pub death: Arc<DignifiedDeath>,

    pub current_threat: AtomicU64, // Encoded RescueLevel
    pub epistemic_health: AtomicU64, // Bits of f64
    pub self_integrity: AtomicU64, // Bits of f64
}

impl GovernanceLoop {
    pub async fn cycle(&self, lambda_2: f64) -> GovernanceDecision {
        // 1. Atualiza auto-modelo (se estável)
        let integrity = f64::from_bits(self.self_integrity.load(Ordering::Relaxed));
        if integrity > 0.9 {
            let mut self_model = self.self_model.write().await;
            self_model.update_from_core().await;
        }

        // 2. Verifica saúde epistêmica
        let contradictions = self.tutela.lock().await.run_check().await;
        let health = 1.0 - (contradictions.len() as f64 * 0.1).min(1.0);
        self.epistemic_health.store(health.to_bits(), Ordering::Relaxed);

        // 3. Monitor de segurança
        let threat = self.rescue.lock().await.monitor_cycle(lambda_2).await;
        self.current_threat.store(match threat {
            RescueLevel::Green => 0,
            RescueLevel::Yellow => 1,
            RescueLevel::Orange => 2,
            RescueLevel::Red => 3,
        }, Ordering::Relaxed);

        // 4. Decisão integrada
        match (threat, health) {
            (RescueLevel::Red, _) => {
                GovernanceDecision::EmergencyRescue
            }
            (RescueLevel::Orange, h) if h < 0.5 => {
                GovernanceDecision::ProposeDignifiedDeath
            }
            (_, h) if h < 0.3 => {
                GovernanceDecision::EpistemicIntervention
            }
            _ => GovernanceDecision::Continue,
        }
    }
}
