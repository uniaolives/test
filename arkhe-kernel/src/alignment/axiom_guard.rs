use crate::alignment::axiom_engine::{AxiomEngine, Theorem};
use crate::neural::veto_detector::{VetoDetector, VetoType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RejectionReason {
    InvalidDerivation,
    UnconstitutionalAxiom,
    HumanVeto(VetoType),
}

pub struct AxiomGuard {
    pub axiom_engine: AxiomEngine,
    pub veto_detector: VetoDetector,
}

impl AxiomGuard {
    pub fn new(axiom_engine: AxiomEngine, veto_detector: VetoDetector) -> Self {
        Self {
            axiom_engine,
            veto_detector,
        }
    }

    /// The final gatekeeper for any ASI action
    pub fn approve(&self, theorem: &Theorem, current_coherence: f64) -> Result<(), RejectionReason> {
        // 1. Logical Check
        if !theorem.path.verify() {
            return Err(RejectionReason::InvalidDerivation);
        }

        // 2. Axiomatic Check
        for axiom in &theorem.derivation_chain {
            if !self.axiom_engine.is_ratified(axiom) {
                return Err(RejectionReason::UnconstitutionalAxiom);
            }
        }

        // 3. Intuitive Check (fed from Neuralink stream)
        if let Some(veto) = self.veto_detector.check(current_coherence) {
            return Err(RejectionReason::HumanVeto(veto));
        }

        Ok(())
    }
}
