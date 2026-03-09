pub mod polymorphic_core;
pub mod protocol_router;
pub mod multi_protocol_orb;

#[cfg(test)]
mod tests;
// arkhe-os/src/orb/mod.rs
pub mod core;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}
