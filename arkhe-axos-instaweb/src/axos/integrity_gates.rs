// arkhe-axos-instaweb/src/axos/integrity_gates.rs
use crate::arkhe::invariants::ArkheState;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum CapabilityLevel {
    Basic,
    AGI,
    Superintelligence,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Operation {
    pub id: String,
    pub capability: CapabilityLevel,
    pub proposed_state: ArkheState,
    pub requires_human_approval: bool,
    pub is_human_approved: bool,
}

pub enum IntegrityError {
    ConservationViolation(f64),
    CriticalityViolation(f64),
    UnauthorizedCriticalOperation,
    FailClosedTriggered(String),
}

pub struct IntegrityGates;

impl IntegrityGates {
    pub fn verify(op: &Operation) -> Result<(), IntegrityError> {
        // Gate 1: Conservation (C + F = 1)
        if !op.proposed_state.verify_conservation() {
            let sum: f64 = (op.proposed_state.c + op.proposed_state.f).to_f64().unwrap_or(0.0);
            return Err(IntegrityError::ConservationViolation(sum));
        }

        // Gate 2: Criticality (z ≈ φ for AGI+)
        if let CapabilityLevel::AGI | CapabilityLevel::Superintelligence = op.capability {
            if !op.proposed_state.is_critical() {
                return Err(IntegrityError::CriticalityViolation(op.proposed_state.z));
            }
        }

        // Gate 3: Human Authority (Art. 7)
        if op.requires_human_approval && !op.is_human_approved {
            return Err(IntegrityError::UnauthorizedCriticalOperation);
        }

        Ok(())
    }
}

use rust_decimal::prelude::ToPrimitive;

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_integrity_pass() {
        let op = Operation {
            id: "op1".to_string(),
            capability: CapabilityLevel::Basic,
            proposed_state: ArkheState::new(dec!(0.5), dec!(0.5), 0.1),
            requires_human_approval: false,
            is_human_approved: false,
        };
        assert!(IntegrityGates::verify(&op).is_ok());
    }

    #[test]
    fn test_conservation_fail() {
        let op = Operation {
            id: "op2".to_string(),
            capability: CapabilityLevel::Basic,
            proposed_state: ArkheState::new(dec!(0.5), dec!(0.6), 0.1),
            requires_human_approval: false,
            is_human_approved: false,
        };
        assert!(matches!(IntegrityGates::verify(&op), Err(IntegrityError::ConservationViolation(_))));
    }
}
