pub mod manifold;
pub use manifold::{GlobalManifold, QuantumState, SelfModification, Node, KrausOperator, KrausChannel};
pub mod kraus;
pub use manifold::{GlobalManifold, QuantumState, KrausOperator, SelfModification, Node};
pub use kraus::KrausChannel;
