use std::time::Duration;

pub struct AutoDecisionSystem {
    pub checkpoints_2_4: CheckpointConfig,
    pub checkpoint_5: FinalCheckpointConfig,
}

pub struct CheckpointConfig {
    pub auto_continue_criteria: &'static str,
    pub pause_criteria: &'static str,
    pub escalation_path: Vec<&'static str>,
    pub max_pause_before_human_decision: Duration,
}

pub struct FinalCheckpointConfig {
    pub architect_review_required: bool,
    pub approval_window: Duration,
    pub default_action_if_silent: &'static str,
    pub fallback_if_rejected: FallbackConfig,
}

pub struct FallbackConfig {
    pub action: &'static str,
    pub rationale: &'static str,
    pub architect_decision_point: &'static str,
}

impl AutoDecisionSystem {
    pub fn new() -> Self {
        Self {
            checkpoints_2_4: CheckpointConfig {
                auto_continue_criteria: "all_metrics > 0.95 * projected_value",
                pause_criteria: "any_metric < 0.90 * projected_value for 60s",
                escalation_path: vec![
                    "1. Pause expansion for 120s automatic analysis",
                    "2. If issue persists, notify Architect_Ω directly",
                    "3. Architect_Ω decides: continue, delay, or regress to 25%",
                    "4. Regression to 25% takes 15 seconds (pre-configured fallback)",
                ],
                max_pause_before_human_decision: Duration::from_secs(300),
            },
            checkpoint_5: FinalCheckpointConfig {
                architect_review_required: true,
                approval_window: Duration::from_secs(300),
                default_action_if_silent: "proceed_to_24h_observation",
                fallback_if_rejected: FallbackConfig {
                    action: "maintain_48pct_for_6h_review",
                    rationale: "Permit review deeper before aborting 50% completely",
                    architect_decision_point: "T+02:11:30",
                },
            },
        }
    }
}
