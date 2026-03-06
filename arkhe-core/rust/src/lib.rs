pub mod ledger;
pub mod constitution;
pub mod ffi;
pub mod nexus_registry;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArkheError {
    #[error("Constitutional violation: {0}")]
    ConstitutionViolation(String),
    #[error("Ledger corruption")]
    LedgerCorruption,
    #[error("Coherence below threshold: {coherence}")]
    CoherenceError { coherence: f64 },
    #[error("FFI error: {0}")]
    FfiError(String),
}

pub type Result<T> = std::result::Result<T, ArkheError>;
