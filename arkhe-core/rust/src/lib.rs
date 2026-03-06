pub mod ledger;
pub mod constitution;
pub mod ffi;
pub mod kernel;
pub mod physics;
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

#[cfg(test)]
mod tests {
    use crate::kernel::scheduler::{CoherenceScheduler, Task};

    #[test]
    fn test_miller_limit_protection() {
        let mut scheduler = CoherenceScheduler::new();

        // Priority 255 will come out first!
        scheduler.submit(Task {
            id: 1,
            description: "Normal Task".to_string(),
            coherence_load: 0.5,
            priority: 10,
        });
        scheduler.submit(Task {
            id: 2,
            description: "Dangerous Task".to_string(),
            coherence_load: 5.0,
            priority: 255,
        });

        // First tick should be the dangerous task
        assert!(scheduler.tick().is_err());
        // Second tick should be the normal task
        assert!(scheduler.tick().is_ok());
    }
}
