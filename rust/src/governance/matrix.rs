// rust/src/governance/matrix.rs

use std::collections::HashMap;

pub struct ProposedAction {
    pub id: String,
    pub content: String,
}

pub struct ActionContext {
    pub user: String,
    pub phi: f64,
}

#[derive(Debug, PartialEq)]
pub enum LayerDecision { Approve, Reject, StrongReject }

#[derive(Debug)]
pub enum GovernanceDecision {
    ConservativeApproval,
    BalancedApproval,
    ProgressiveApproval,
    SymbioticApproval,
    Reject(String),
    EmergencyOverride(EmergencyProtocol),
}

#[derive(Debug)]
pub struct EmergencyProtocol {
    pub name: String,
}

pub struct ArchetypeGovernance;
impl ArchetypeGovernance {
    pub fn evaluate(&self, _action: &ProposedAction) -> LayerDecision { LayerDecision::Approve }
}

pub struct EthicalCommittee;
impl EthicalCommittee {
    pub fn evaluate(&self, _action: &ProposedAction, _ctx: &ActionContext) -> LayerDecision { LayerDecision::Approve }
}

pub struct TechnicalGovernance;
impl TechnicalGovernance {
    pub fn evaluate(&self, _action: &ProposedAction) -> LayerDecision { LayerDecision::Approve }
}

pub struct UserGovernance;
impl UserGovernance {
    pub fn evaluate(&self, _action: &ProposedAction, _user: &str) -> LayerDecision { LayerDecision::Approve }
}

pub struct PhiStateMachine;
pub struct EmergencyProtocolRegistry;
impl EmergencyProtocolRegistry {
    pub fn get_protocol(&self, _action: &ProposedAction) -> EmergencyProtocol {
        EmergencyProtocol { name: "DEFAULT_EMERGENCY".to_string() }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum GovernanceLevel {
    Archetype,
    Ethical,
    Technical,
    User,
}

pub struct CompleteGovernanceMatrix {
    pub archetype_layer: ArchetypeGovernance,
    pub ethical_layer: EthicalCommittee,
    pub technical_layer: TechnicalGovernance,
    pub user_layer: UserGovernance,
    pub _phi_state_machine: PhiStateMachine,
    pub emergency_protocols: EmergencyProtocolRegistry,
}

impl CompleteGovernanceMatrix {
    pub fn evaluate_action(
        &self,
        action: &ProposedAction,
        context: &ActionContext,
    ) -> GovernanceDecision {
        let mut decisions = HashMap::new();

        decisions.insert(GovernanceLevel::Archetype, self.archetype_layer.evaluate(action));
        decisions.insert(GovernanceLevel::Ethical, self.ethical_layer.evaluate(action, context));
        decisions.insert(GovernanceLevel::Technical, self.technical_layer.evaluate(action));
        decisions.insert(GovernanceLevel::User, self.user_layer.evaluate(action, &context.user));

        if self.detect_emergency_condition(action, context) {
            return GovernanceDecision::EmergencyOverride(
                self.emergency_protocols.get_protocol(action)
            );
        }

        self.apply_phi_weighting(decisions, context.phi)
    }

    fn apply_phi_weighting(
        &self,
        decisions: HashMap<GovernanceLevel, LayerDecision>,
        phi: f64,
    ) -> GovernanceDecision {
        match phi {
            p if p < 0.65 => {
                if decisions[&GovernanceLevel::Archetype] == LayerDecision::Reject {
                    GovernanceDecision::Reject("Archetype violation".into())
                } else {
                    GovernanceDecision::ConservativeApproval
                }
            }
            p if p < 0.78 => {
                let rejections = decisions.values()
                    .filter(|&d| *d == LayerDecision::Reject || *d == LayerDecision::StrongReject)
                    .count();

                if rejections >= 2 {
                    GovernanceDecision::Reject("Multiple layer rejections".into())
                } else {
                    GovernanceDecision::BalancedApproval
                }
            }
            p if p < 0.85 => {
                if decisions[&GovernanceLevel::User] == LayerDecision::Approve &&
                   decisions[&GovernanceLevel::Ethical] != LayerDecision::StrongReject {
                    GovernanceDecision::ProgressiveApproval
                } else {
                    GovernanceDecision::Reject("User consent missing or strong ethical rejection".into())
                }
            }
            _ => {
                if decisions.values().all(|d| *d == LayerDecision::Approve) {
                    GovernanceDecision::SymbioticApproval
                } else {
                    GovernanceDecision::Reject("Not all layers approve for symbiotic mode".into())
                }
            }
        }
    }

    fn detect_emergency_condition(&self, _action: &ProposedAction, _ctx: &ActionContext) -> bool {
        false
    }
}
