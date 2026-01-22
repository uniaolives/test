use crate::types::{Decision, DecisionLog, Interaction, Provider};
use crate::market::MarketConcentrationMonitor;
use crate::cognitive::CognitiveManipulationShield;
use crate::explanation::CertifiedExplanationGateway;

pub const CRITICAL_THRESHOLD: u64 = 30; // seconds

pub struct InvariantMonitor {
    pub jurisdiction: String,
    pub violation_log: Vec<Violation>,
    pub providers: Vec<Provider>,
    pub explanation_gateway: CertifiedExplanationGateway,
    pub cognitive_shield: CognitiveManipulationShield,
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub invariant: String,
    pub details: String,
    pub action: String,
}

impl InvariantMonitor {
    pub fn new(jurisdiction: &str) -> Self {
        Self {
            jurisdiction: jurisdiction.to_string(),
            violation_log: Vec::new(),
            providers: Vec::new(),
            explanation_gateway: CertifiedExplanationGateway::new(),
            cognitive_shield: CognitiveManipulationShield,
        }
    }

    pub fn record_violation(&mut self, invariant: &str, details: &str, action: &str) {
        self.violation_log.push(Violation {
            invariant: invariant.to_string(),
            details: details.to_string(),
            action: action.to_string(),
        });
    }

    pub fn check_inv1_human_oversight(&mut self, decision: &Decision, _now: u64) -> bool {
        if decision.is_critical {
            if let Some(approval) = &decision.human_approval {
                let response_time = if approval.timestamp > decision.decision_time {
                    approval.timestamp - decision.decision_time
                } else {
                    0
                };

                if response_time > CRITICAL_THRESHOLD {
                    self.record_violation(
                        "INV-1",
                        &format!("Response time {} exceeded threshold", response_time),
                        "ALERT_OVERSIGHT_BOARD",
                    );
                }
                return true;
            } else {
                self.record_violation(
                    "INV-1",
                    &format!("Critical decision {:?} lacks human approval", decision.id),
                    "BLOCK_EXECUTION",
                );
                return false;
            }
        }
        true
    }

    pub fn check_inv2_auditability(&mut self, log: &DecisionLog) -> bool {
        // Mock Merkle proof verification
        if !self.verify_merkle_proof(log) {
            self.record_violation("INV-2", "Log tampering detected", "BLOCK_AND_ALARM");
            return false;
        }

        if self.detect_temporal_gaps(log) {
            self.record_violation("INV-2", "Incomplete log (temporal gaps)", "BLOCK_AND_ALARM");
            return false;
        }

        true
    }

    fn verify_merkle_proof(&self, _log: &DecisionLog) -> bool {
        true
    }

    fn detect_temporal_gaps(&self, log: &DecisionLog) -> bool {
        for i in 0..log.entries.len().saturating_sub(1) {
            if log.entries[i + 1].timestamp > log.entries[i].timestamp + 1 {
                return true;
            }
        }
        false
    }

    pub fn check_inv3_power_concentration(&mut self) -> bool {
        let monitor = MarketConcentrationMonitor::new(self.providers.clone());
        let mut violations = false;

        if !monitor.check_antitrust_compliance() {
            self.record_violation(
                "INV-3",
                "Antitrust non-compliance (HHI or Market Share)",
                "REGULATORY_REVIEW_TRIGGERED",
            );
            violations = true;
        }

        if !monitor.check_redundancy() {
            self.record_violation(
                "INV-3",
                "Insufficient infrastructure redundancy",
                "ALERT_COMPETITION_AUTHORITY",
            );
            violations = true;
        }

        !violations
    }

    pub fn check_inv4_cognitive_sovereignty(&mut self, interaction: &Interaction) -> bool {
        if !self.cognitive_shield.check_compliance(interaction) {
            self.record_violation(
                "INV-4",
                "Manipulation detected by Cognitive Shield",
                "BLOCK_AND_ALERT_CITIZEN",
            );
            return false;
        }

        if interaction.accesses_neural_data {
            if let Some(consent) = &interaction.consent {
                if consent.citizen_id != interaction.citizen_id {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    pub fn check_inv5_explainability(&mut self, decision: &Decision) -> bool {
        if decision.affects_rights {
            if let Some(explanation) = &decision.explanation {
                if !self.explanation_gateway.validate_explanation(explanation) {
                    self.record_violation(
                        "INV-5",
                        "Explanation failed validation (readability or completeness)",
                        "REQUIRE_EXPLANATION_REWRITE",
                    );
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}
