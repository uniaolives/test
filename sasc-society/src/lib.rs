pub mod google {
    pub mod protobuf {
        pub use prost_types::{Timestamp, Duration};
    }
}

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
pub mod grpc;

pub mod vajra_sasc_bridge;
pub mod hsm_signer;
pub mod constants;
pub mod gaia_integration;
