pub mod manifold;
pub use manifold::{GlobalManifold, QuantumState, SelfModification, Node, KrausOperator};
pub mod kraus;
pub use kraus::KrausChannel;
