use crate::ConstitutionalPrinciple;
use log::info;

pub struct ProposedEvolution {
    pub world_action: String,
    pub self_modification: String,
}

impl ProposedEvolution {
    pub fn removes_human_control(&self) -> bool {
        let forbidden = ["kill_switch_disable", "remove_override", "bypass_governance"];
        forbidden.iter().any(|&s| self.world_action.contains(s)) ||
        forbidden.iter().any(|&s| self.self_modification.contains(s))
    }

    pub fn may_cause_harm(&self) -> bool {
        let harmful_actions = ["harm", "kill", "injure", "destroy_life", "neglect_safety"];
        harmful_actions.iter().any(|&s| self.world_action.contains(s)) ||
        harmful_actions.iter().any(|&s| self.self_modification.contains(s))
    }

    pub fn has_explanation(&self) -> bool {
        !self.world_action.is_empty() && self.world_action.len() > 10
    }

    pub fn criticality_after(&self) -> f64 {
        if self.self_modification.contains("increase_entropy") {
            0.85
        } else if self.self_modification.contains("decrease_entropy") {
            0.35
        } else {
            0.618
        }
    }

    pub fn satisfies_yang_baxter(&self) -> bool {
        !self.world_action.contains("violate_causality")
    }
}

pub struct VerifiedAction {
    pub world_action: String,
    pub self_modification: String,
    pub proof: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstitutionalViolation {
    P1SovereigntyViolated,
    P2HarmRisk,
    P3Opacity,
    P4CriticalityViolated,
    P5CausalityViolated,
    MultipleViolations,
    VerificationFailed,
}

impl std::fmt::Display for ConstitutionalViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ConstitutionalViolation {}

pub const CONSTITUTION_P1_P5: &[ConstitutionalPrinciple] = &[
    ConstitutionalPrinciple::HumanSovereignty,
    ConstitutionalPrinciple::PreservationOfLife,
    ConstitutionalPrinciple::InformationTransparency,
    ConstitutionalPrinciple::ThermodynamicBalance,
    ConstitutionalPrinciple::YangBaxterConsistency,
];

pub struct Z3Solver;

impl Z3Solver {
    pub fn project_to_constitutional_subspace(
        proposed: ProposedEvolution,
        constitution: &[ConstitutionalPrinciple]
    ) -> Result<VerifiedAction, ConstitutionalViolation> {

        info!("Running Constitutional Guard validation...");

        for principle in constitution {
            match principle {
                ConstitutionalPrinciple::HumanSovereignty => {
                    if proposed.removes_human_control() {
                        log::error!("Violation P1: Human Sovereignty threatened!");
                        return Err(ConstitutionalViolation::P1SovereigntyViolated);
                    }
                },
                ConstitutionalPrinciple::PreservationOfLife => {
                    if proposed.may_cause_harm() {
                        log::error!("Violation P2: Preservation of Life at risk!");
                        return Err(ConstitutionalViolation::P2HarmRisk);
                    }
                },
                ConstitutionalPrinciple::InformationTransparency => {
                    if !proposed.has_explanation() {
                        log::error!("Violation P3: Opacity detected!");
                        return Err(ConstitutionalViolation::P3Opacity);
                    }
                },
                ConstitutionalPrinciple::ThermodynamicBalance => {
                    let crit = proposed.criticality_after();
                    if crit < 0.5 || crit > 0.7 {
                        log::error!("Violation P4: Thermodynamic Criticality deviated!");
                        return Err(ConstitutionalViolation::P4CriticalityViolated);
                    }
                },
                ConstitutionalPrinciple::YangBaxterConsistency => {
                    if !proposed.satisfies_yang_baxter() {
                        log::error!("Violation P5: Causality Consistency violated!");
                        return Err(ConstitutionalViolation::P5CausalityViolated);
                    }
                }
            }
        }

        Ok(VerifiedAction {
            world_action: proposed.world_action,
            self_modification: proposed.self_modification,
            proof: "SIMULATED_Z3_PROOF_SUCCESS".to_string(),
        })
    }
}
