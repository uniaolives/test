// rust/src/bio_layer/dopamine_monitor.rs

use std::time::Duration;

pub struct DopamineSpike {
    pub intensity: f64,
}

pub struct RewardEvent {
    pub intensity: f64,
}

pub struct ModifiedReward {
    pub intensity: f64,
    pub delay_ms: u64,
    pub requires_physical_movement: bool,
}

pub struct DopamineSatietyMonitor {
    pub activation_history: Vec<(std::time::Instant, DopamineSpike)>,
    pub current_satiety_threshold: f64,
}

impl DopamineSatietyMonitor {
    pub fn apply_friction(&mut self, proposed_reward: RewardEvent) -> ModifiedReward {
        let now = std::time::Instant::now();
        let recent_activation: f64 = self.activation_history
            .iter()
            .filter(|(t, _)| now.duration_since(*t) < Duration::from_secs(3600))
            .map(|(_, s)| s.intensity)
            .sum();

        if recent_activation > self.current_satiety_threshold {
            return ModifiedReward {
                intensity: proposed_reward.intensity * 0.5,
                delay_ms: 2000,
                requires_physical_movement: true,
            };
        }

        ModifiedReward {
            intensity: proposed_reward.intensity,
            delay_ms: 0,
            requires_physical_movement: false,
        }
    }
}
