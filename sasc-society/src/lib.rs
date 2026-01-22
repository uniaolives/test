pub mod engine;
pub mod agents;
pub mod audit;
pub mod integration;

#[cfg(test)]
mod tests;

pub use engine::diversity::PerspectiveDiversityEngine;
pub use engine::dialectic::DialecticSynthesizer;
pub use agents::role::SocioEmotionalRole;
pub use audit::provenance::ProvenanceTracer;
