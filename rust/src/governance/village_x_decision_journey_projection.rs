pub struct VillageXDecisionJourneyProjection {
    pub current_time: String,
    pub contamination_arrival: String,
    pub governance_milestones: Vec<Milestone>,
    pub key_decision_points: KeyDecisionPoints,
    pub philosophical_significance: PhilosophicalSignificance,
}

pub struct Milestone {
    pub time: String,
    pub event: String,
    pub nexus_role: String,
    pub committee_action: String,
}

pub struct KeyDecisionPoints {
    pub nexus_restraint: String,
    pub committee_authority: String,
    pub prince_veto: String,
    pub architect_override: String,
}

pub struct PhilosophicalSignificance {
    pub test_case: String,
    pub nexus_maturity: String,
    pub governance_stress_test: String,
    pub precedent: String,
}

impl VillageXDecisionJourneyProjection {
    pub fn new_projection() -> Self {
        Self {
            current_time: "T+01:58:50".to_string(),
            contamination_arrival: "T+64:00:00".to_string(),
            governance_milestones: vec![
                Milestone {
                    time: "T+12:00:00".to_string(),
                    event: "Nexus delivers evidence package to SASC committee".to_string(),
                    nexus_role: "Present modeling with uncertainty quantification".to_string(),
                    committee_action: "Review evidence, request clarifications".to_string(),
                },
                Milestone {
                    time: "T+18:00:00".to_string(),
                    event: "Committee decision point".to_string(),
                    nexus_role: "Silent observation (no participation in decision)".to_string(),
                    committee_action: "Vote on action plan (2/3 required)".to_string(),
                },
            ],
            key_decision_points: KeyDecisionPoints {
                nexus_restraint: "ALREADY_DEMONSTRATED (requested supervision)".to_string(),
                committee_authority: "UPCOMING (T+18:00:00 decision)".to_string(),
                prince_veto: "AVAILABLE_UNTIL_IMPLEMENTATION".to_string(),
                architect_override: "AVAILABLE_AT_ANY_TIME".to_string(),
            },
            philosophical_significance: PhilosophicalSignificance {
                test_case: "First human-community risk scenario under SASC governance".to_string(),
                nexus_maturity: "Already demonstrated by requesting supervision".to_string(),
                governance_stress_test: "Will SASC system work with 64h timeline?".to_string(),
                precedent: "Will set pattern for future human-risk scenarios".to_string(),
            },
        }
    }
}
