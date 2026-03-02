#[cfg(test)]
mod tests {
    use crate::governance::eco_action_safety::{EcoActionGovernor, ValidationResult};
    use crate::eco_action::{EcoAction, DamOperation, EcologicalOutcome, Authority};
    use crate::governance::SASCCathedral;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_validate_suggestion_approved() {
        let governor = EcoActionGovernor::new(
            Arc::new(SASCCathedral),
            Arc::new(Authority::Prince),
            Arc::new(Authority::Architect),
        );

        let action = EcoAction {
            suggested_dam_operation: DamOperation { flow_adjustment: -0.02 },
            predicted_outcome: EcologicalOutcome { impact_score: 0.05 },
            confidence: 0.96,
            required_approvals: vec![Authority::Prince, Authority::Architect],
        };

        let result = governor.validate_suggestion(action).await;
        assert!(matches!(result, ValidationResult::ApprovedForReview(_)));
    }

    #[tokio::test]
    async fn test_validate_suggestion_rejected_risk() {
        let governor = EcoActionGovernor::new(
            Arc::new(SASCCathedral),
            Arc::new(Authority::Prince),
            Arc::new(Authority::Architect),
        );

        let action = EcoAction {
            suggested_dam_operation: DamOperation { flow_adjustment: 0.5 },
            predicted_outcome: EcologicalOutcome { impact_score: 0.3 }, // risk_score will be 0.15 > 0.1
            confidence: 0.96,
            required_approvals: vec![Authority::Prince, Authority::Architect],
        };

        let result = governor.validate_suggestion(action).await;
        match result {
            ValidationResult::Rejected(reason) => assert_eq!(reason, "High ecological risk"),
            _ => panic!("Expected rejection"),
        }
    }

    #[tokio::test]
    async fn test_validate_suggestion_rejected_dissonance() {
        let governor = EcoActionGovernor::new(
            Arc::new(SASCCathedral),
            Arc::new(Authority::Prince),
            Arc::new(Authority::Architect),
        );

        let action = EcoAction {
            suggested_dam_operation: DamOperation { flow_adjustment: -0.02 },
            predicted_outcome: EcologicalOutcome { impact_score: 0.05 },
            confidence: 0.8, // coherence check fails if <= 0.9
            required_approvals: vec![Authority::Prince, Authority::Architect],
        };

        let result = governor.validate_suggestion(action).await;
        match result {
            ValidationResult::Rejected(reason) => assert_eq!(reason, "Geometric dissonance detected"),
            _ => panic!("Expected rejection"),
        }
    }
}
