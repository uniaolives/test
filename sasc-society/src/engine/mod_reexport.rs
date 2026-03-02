use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoTDecisionRequest {
    pub problem_statement: String,
    pub constitutional_context: Vec<String>,
    pub constraints: Vec<crate::engine::Constraint>,
    pub stakeholders: Vec<crate::engine::Stakeholder>,
    pub deadline: Option<std::time::SystemTime>,
    pub priority: u8,
    pub metadata: crate::engine::DecisionMetadata,
}
