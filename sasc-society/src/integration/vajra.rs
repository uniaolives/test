use blake3::Hash;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VajraAlert {
    PerspectiveActivated {
        persona_id: crate::agents::PersonaId,
        problem_hash: [u8; 32],
        activation_hash: [u8; 32],
    },
    DiversityAssessment {
        score: f64,
        active_perspectives: usize,
        dominance_share: f64,
    },
    DecisionFinalized {
        decision_id: [u8; 32],
        status: String,
        coherence: f64,
        perspectives_used: usize,
        processing_time_ms: u128,
        phi_trajectory: Vec<f64>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Success,
    Warning,
    Critical,
    Error,
}

pub fn report_to_vajra(alert: VajraAlert, severity: AlertSeverity) {
    // Dummy implementation
    log::info!("Vajra Report [{:?}]: {:?}", severity, alert);
}
