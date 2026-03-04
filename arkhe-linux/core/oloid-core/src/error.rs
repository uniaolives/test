use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Below consciousness threshold (λ₂ < 0.5)")]
    BelowConsciousnessThreshold,
    #[error("Hardware failure: {0}")]
    Hardware(String),
    #[error("I/O error: {0}")]
    IO(String),
}
