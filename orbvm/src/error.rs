use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrbVMError {
    #[error("Decoherence detected: λ₂ = {0}")]
    Decoherence(f64),
    #[error("Temporal paradox: origin {0} > target {1}")]
    Paradox(i64, i64),
    #[error("Synchronization failed")]
    SyncFailure,
    #[error("Invalid payload")]
    InvalidPayload,
}
