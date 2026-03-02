use thiserror::Error;

#[derive(Error, Debug)]
pub enum ASIError {
    #[error("Too many structures loaded: {current} > {max}")]
    TooManyStructures { current: usize, max: usize },

    #[error("No structures available for processing")]
    NoStructuresAvailable,

    #[error("Empty composition input")]
    EmptyCompositionInput,

    #[error("Invariant validation failed: {invariant} - {reason}")]
    InvariantValidationFailed { invariant: String, reason: String },

    #[error("Time limit exceeded: {elapsed:?} > {limit:?}")]
    TimeLimitExceeded { elapsed: std::time::Duration, limit: std::time::Duration },

    #[error("State integrity check failed: {0}")]
    StateIntegrityFailed(String),

    #[error("Generic error: {0}")]
    Generic(String),

    #[error("Web777 error: {0}")]
    Web777(String),

    #[error("Bridge error: {0}")]
    Bridge(String),
}
