pub enum VerificationContext {
    TruthSubmission,
}

pub struct AttestationStatus {
    hard_frozen: bool,
    agent_id: String,
    phi_weight: f64,
}

impl AttestationStatus {
    pub fn new(hard_frozen: bool, agent_id: &str, phi_weight: f64) -> Self {
        Self {
            hard_frozen,
            agent_id: agent_id.to_string(),
            phi_weight,
        }
    }

    pub fn is_hard_frozen(&self) -> bool {
        self.hard_frozen
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn consciousness_weight(&self) -> f64 {
        self.phi_weight
    }
}

pub struct PhiThreshold(pub f64);
