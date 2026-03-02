pub mod z3_verifier;

pub use z3_verifier::{Z3Solver, ConstitutionalViolation, CONSTITUTION_P1_P5, ProposedEvolution, VerifiedAction};

#[derive(Debug, Clone, Copy)]
pub enum ConstitutionalPrinciple {
    HumanSovereignty,      // P1
    PreservationOfLife,    // P2
    InformationTransparency, // P3
    ThermodynamicBalance,  // P4
    YangBaxterConsistency, // P5
}
