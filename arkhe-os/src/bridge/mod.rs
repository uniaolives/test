pub mod impossible;
// arkhe-os/src/bridge/mod.rs
pub mod tcpip;
pub mod rf;
pub mod blockchain;
pub mod industrial;
pub mod mesh;
pub mod dark;
pub mod universal_router;

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Orb error: {0}")]
    Orb(#[from] crate::orb::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Blockchain error: {0}")]
    Blockchain(String),
    #[error("Tor error: {0}")]
    Tor(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
