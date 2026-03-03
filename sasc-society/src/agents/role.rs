use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocioEmotionalRole {
    pub openness: f64,
    pub conscientiousness: f64,
    pub extraversion: f64,
    pub agreeableness: f64,
    pub neuroticism: f64,
}

impl SocioEmotionalRole {
    pub fn default_empathy() -> Self {
        Self {
            openness: 0.8,
            conscientiousness: 0.9,
            extraversion: 0.5,
            agreeableness: 0.9,
            neuroticism: 0.2,
        }
    }
}

impl Default for SocioEmotionalRole {
    fn default() -> Self {
        Self::default_empathy()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersonaRole {
    Analyst,
    Guardian,
    Liaison,
    Physicist,
}
