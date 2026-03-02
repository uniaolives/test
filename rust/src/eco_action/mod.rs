#[derive(Debug, Clone)]
pub struct DamOperation {
    pub flow_adjustment: f64,
}

#[derive(Debug, Clone)]
pub struct EcologicalOutcome {
    pub impact_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authority {
    Sasc,
    Prince,
    Architect,
}

pub enum ExecutionResult {
    AwaitingApproval,
    Success,
}

pub struct EcoAction {
    pub suggested_dam_operation: DamOperation,
    pub predicted_outcome: EcologicalOutcome,
    pub confidence: f64,
    pub required_approvals: Vec<Authority>,
}

impl EcoAction {
    pub async fn execute_if_approved(&self) -> ExecutionResult {
        ExecutionResult::AwaitingApproval
    }

    pub fn is_geometrically_coherent(&self) -> bool {
        // Placeholder check: ações com confiança > 0.9 são consideradas coerentes
        self.confidence > 0.9
    }
}
