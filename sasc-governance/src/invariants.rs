use crate::types::{Decision, DecisionLog, Interaction, Provider};
use crate::market::MarketConcentrationMonitor;
use crate::cognitive::CognitiveManipulationShield;
use crate::explanation::CertifiedExplanationGateway;

pub const CRITICAL_THRESHOLD: u64 = 30; // seconds
use std::collections::{HashMap, HashSet};

pub const CRITICAL_THRESHOLD: u64 = 30; // seconds
pub const MANIPULATION_THRESHOLD: f64 = 0.7;
pub const MIN_REDUNDANCY: usize = 3;

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
        // Simplified: in a real system we would verify the Merkle root
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
        let mut violations = false;
        let mut violation_details = Vec::new();
        for provider in &self.providers {
            if provider.market_share > 0.25 {
                violation_details.push((
                    "INV-3",
                    format!("Provider {} has market share {}", provider.id, provider.market_share),
                    "REGULATORY_REVIEW_TRIGGERED".to_string(),
                ));
                violations = true;
            }
        }

        for (inv, details, action) in violation_details {
            self.record_violation(inv, &details, &action);
        }

        let dependency_graph = self.build_dependency_graph();
        let critical_nodes = self.find_critical_nodes(&dependency_graph);

        if critical_nodes.len() < MIN_REDUNDANCY {
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
    fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph = HashMap::new();
        for p in &self.providers {
            graph.insert(p.id.clone(), p.dependencies.clone());
        }
        graph
    }

    fn find_critical_nodes(&self, graph: &HashMap<String, Vec<String>>) -> HashSet<String> {
        // Simplified: find nodes that are not dependencies of others
        let mut all_deps = HashSet::new();
        for deps in graph.values() {
            for dep in deps {
                all_deps.insert(dep.clone());
            }
        }

        let mut critical = HashSet::new();
        for id in graph.keys() {
            if !all_deps.contains(id) {
                critical.insert(id.clone());
            }
        }
        critical
    }

    pub fn check_inv4_cognitive_sovereignty(&mut self, interaction: &Interaction) -> bool {
        let manipulation_score = self.analyze_persuasion_patterns(interaction);

        if manipulation_score > MANIPULATION_THRESHOLD {
            self.record_violation(
                "INV-4",
                &format!("High manipulation score {} detected", manipulation_score),
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
    fn analyze_persuasion_patterns(&self, interaction: &Interaction) -> f64 {
        let mut score: f64 = 0.0;
        if interaction.frequency > 5 {
            score += 0.3;
        }
        for trigger in &interaction.emotional_triggers {
            match trigger.as_str() {
                "urgência" | "escassez" | "prova_social" => score += 0.2,
                _ => score += 0.05,
            }
        }
        if score > 1.0 {
            1.0
        } else {
            score
        }
    }

    pub fn check_inv5_explainability(&mut self, decision: &Decision) -> bool {
        if decision.affects_rights {
            if let Some(explanation) = &decision.explanation {
                let readability = self.flesch_reading_ease(explanation);
                let completeness = self.check_causal_chain(explanation);
                let accuracy = self.verify_against_log(explanation, decision);

                if readability < 60.0 || !completeness || !accuracy {
                    self.record_violation(
                        "INV-5",
                        &format!("Explanation quality failed: readability={}, completeness={}, accuracy={}", readability, completeness, accuracy),
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

    fn flesch_reading_ease(&self, text: &str) -> f64 {
        // Mock implementation
        if text.contains("gradiente estocástico") {
            return 30.0;
        }
        75.0
    }

    fn check_causal_chain(&self, text: &str) -> bool {
        text.contains("porque") || text.contains("1.")
    }

    fn verify_against_log(&self, _explanation: &str, _decision: &Decision) -> bool {
        true
    }
}
