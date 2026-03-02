// src/integration/human_echo_approval.rs
use crate::EchoContext;
use crate::EchoError;

pub struct HumanEchoApproval;
pub struct HumanApproval;

impl HumanEchoApproval {
    pub async fn confirm_output(&self, _message: &str, _context: &EchoContext) -> Result<HumanApproval, EchoError> {
        Ok(HumanApproval)
    }
}
