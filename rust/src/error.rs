// rust/src/error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResilientError {
    // Erros de wallet e identidade
    #[error("Wallet error: {0}")]
    Wallet(String),

    #[error("Key management error: {0}")]
    KeyManagement(String),

    #[error("Insufficient funds for storage: needed {needed}, have {have}")]
    InsufficientFunds { needed: u64, have: u64 },

    // Erros de estado
    #[error("State validation failed: {0}")]
    StateValidation(String),

    #[error("State compression error: {0}")]
    StateCompression(String),

    #[error("State size exceeds limit: {size} > {limit}")]
    StateSizeLimit { size: usize, limit: usize },

    // Erros de constituição CGE
    #[error("CGE invariant violation: {invariant} - {reason}")]
    InvariantViolation { invariant: String, reason: String },

    #[error("Torsion threshold exceeded: {current} > {max}")]
    TorsionExceeded { current: f64, max: f64 },

    #[error("Constitution version mismatch: {expected} != {actual}")]
    VersionMismatch { expected: String, actual: String },

    // Erros de checkpoint
    #[error("Checkpoint failed: {0}")]
    Checkpoint(String),

    #[error("Scrubbing error: {0}")]
    Scrubbing(String),

    #[error("Upload strategy error: {0}")]
    UploadStrategy(String),

    // Erros de rede
    #[error("Arweave network error: {0}")]
    ArweaveNetwork(String),

    #[error("Nostr network error: {0}")]
    NostrNetwork(String),

    #[error("Relay connection failed: {0}")]
    RelayConnection(String),

    // Erros de runtime
    #[error("Runtime backend error: {0}")]
    RuntimeBackend(String),

    #[error("Context window exhausted")]
    ContextExhausted,

    // Erros de sistema
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

// Tipo de resultado padronizado
pub type ResilientResult<T> = Result<T, ResilientError>;
