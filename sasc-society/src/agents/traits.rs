use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PersonaId(String);

impl From<&str> for PersonaId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for PersonaId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl PersonaId {
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseDomain {
    Mathematics,
    Ethics,
    Biology,
    Physics,
    SocialSciences,
}

impl Default for ExpertiseDomain {
    fn default() -> Self {
        Self::Ethics
    }
}
