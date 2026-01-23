pub struct DamOperation;
pub struct EcologicalOutcome;

pub enum Authority {
    Sasc,
    Prince,
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
}
