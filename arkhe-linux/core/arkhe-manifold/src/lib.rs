pub mod manifold;
pub mod kraus;
pub use manifold::{GlobalManifold, QuantumState, KrausOperator, SelfModification, Node};
pub use kraus::KrausChannel;
