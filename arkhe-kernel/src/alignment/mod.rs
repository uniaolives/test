pub mod axiom_engine;
pub mod constitutional_axioms;
pub mod derivation_space;
pub mod zk_noir;
pub mod axiom_lifecycle;
pub mod axiom_guard;

pub use axiom_engine::{Axiom, Theorem, AxiomEngine, AlignmentError};
pub use constitutional_axioms::initialize_constitutional_axioms;
pub use derivation_space::{ThoughtVector, DerivationPath};
pub use axiom_lifecycle::{AxiomProposal, Vote, AxiomStatus};
pub use axiom_guard::{AxiomGuard, RejectionReason};
