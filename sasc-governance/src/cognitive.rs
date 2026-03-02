use crate::types::Interaction;

pub const MANIPULATION_THRESHOLD: f64 = 0.3;

pub struct CognitiveManipulationShield;

impl CognitiveManipulationShield {
    pub fn analyze_interaction(&self, interaction: &Interaction) -> f64 {
        let mut score = 0.0;

        // Frequency analysis
        if interaction.frequency > 10 {
            score += 0.25;
        }

        // Emotional triggers analysis
        let triggers_count = interaction.emotional_triggers.len();
        if triggers_count > 3 {
            score += 0.30;
        } else {
            score += triggers_count as f64 * 0.10;
        }

        // Urgency patterns
        for msg in &interaction.messages {
            let msg_lower = msg.to_lowercase();
            if msg_lower.contains("agora") || msg_lower.contains("última chance") || msg_lower.contains("urgente") {
                score += 0.20;
                break;
            }
        }

        // Social proof validation (Mock)
        for msg in &interaction.messages {
            if msg.contains("todos já compraram") || msg.contains("amigos já") {
                score += 0.25;
                break;
            }
        }

        score.min(1.0)
    }

    pub fn check_compliance(&self, interaction: &Interaction) -> bool {
        let score = self.analyze_interaction(interaction);
        score < MANIPULATION_THRESHOLD
    }
}
