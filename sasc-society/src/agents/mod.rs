pub mod role;
pub mod traits;

pub use role::SocioEmotionalRole;
pub use traits::{PersonaId, ExpertiseDomain};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PersonaId(pub [u8; 32]);

impl PersonaId {
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self(*bytes)
    }
}

impl fmt::Display for PersonaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Persona({})", hex::encode(&self.0[..4]))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub id: PersonaId,
    pub role: SocioEmotionalRole,
    pub expertise: Vec<ExpertiseDomain>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocioEmotionalRole {
    Analytic,
    Empathic,
    Creative,
    Stoic,
    Diplomatic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseDomain {
    Ethics,
    Technoscience,
    Legislation,
    Sociology,
    Environment,
}
