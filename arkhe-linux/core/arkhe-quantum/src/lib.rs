pub mod asi_core;
pub mod ledger;
pub mod depin;
pub mod fep_solver;
pub mod manifold_ext;

pub use arkhe_thermodynamics::{VariationalFreeEnergy, Criticality, InternalModel};
pub use arkhe_manifold::{GlobalManifold, QuantumState, KrausOperator, SelfModification, KrausChannel, Node};
pub use arkhe_constitution::{Z3Solver, CONSTITUTION_P1_P5, ProposedEvolution, VerifiedAction};
pub use arkhe_time_crystal::PHI;
pub use arkhe_noether::{ConservedCurrent, SymmetryGenerator};

pub use manifold_ext::ExtendedManifold;

#[cfg(test)]
mod tests;

use log::{info, warn, debug};
