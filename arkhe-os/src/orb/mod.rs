// arkhe-os/src/orb/mod.rs
pub mod core;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}
