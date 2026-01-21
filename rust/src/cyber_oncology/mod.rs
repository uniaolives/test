pub mod protocol;
pub mod metrics;
pub mod immunity;

pub use protocol::*;
pub use metrics::*;
pub use immunity::*;

#[derive(Debug, Clone)]
pub enum RemissionStatus {
    Complete,
    Partial,
    Refractory,
}

#[derive(Debug, Clone)]
pub struct AttackVector {
    pub signature: String,
}

impl AttackVector {
    pub fn coordinated(_gateway: &str, _gkp: &str, _phi: &str) -> Self {
        Self {
            signature: "metastatic_attack_0xALPHA".to_string(),
        }
    }
}

pub struct AttackSequence {
    pub signature: String,
}

impl AttackSequence {
    pub fn signature(&self) -> String {
        self.signature.clone()
    }

    pub fn topology(&self) -> String {
        "quantum_torus".to_string()
    }
}

pub struct Antibody;
impl Antibody {
    pub fn design_from_geometry(_topology: String) -> Self {
        Antibody
    }
}
