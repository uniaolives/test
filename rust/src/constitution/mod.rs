// rust/src/constitution/mod.rs
mod invariants;
mod torsional_analysis;
mod versioning;

pub use invariants::{CGEInvariant, InvariantSet};
pub use torsional_analysis::{TorsionAnalyzer, TorsionMetrics};
pub use versioning::{ConstitutionVersion, VersionManager};
use crate::error::{ResilientError, ResilientResult};
use crate::state::ResilientState;

pub struct CGEValidator {
    invariants: InvariantSet,
    max_torsion: f64,
    version_manager: VersionManager,
    last_valid_state_hash: Option<String>,
}

impl CGEValidator {
    pub fn new() -> Self {
        Self {
            invariants: InvariantSet::default(),
            max_torsion: 0.3,
            version_manager: VersionManager::new("CGE-v1.0".to_string()),
            last_valid_state_hash: None,
        }
    }

    pub fn validate_state(&mut self, state: &ResilientState) -> ResilientResult<()> {
        self.version_manager.validate_version(&state.version)?;

        let invariant_results = self.invariants.check_all(state);
        for result in invariant_results {
            if !result.passed {
                return Err(ResilientError::InvariantViolation {
                    invariant: result.name,
                    reason: result.details,
                });
            }
        }

        if let Some(last_hash) = &self.last_valid_state_hash {
            let torsion = self.calculate_torsion(last_hash, &state.state_hash)?;
            if torsion > self.max_torsion {
                return Err(ResilientError::TorsionExceeded {
                    current: torsion,
                    max: self.max_torsion,
                });
            }
        }

        self.last_valid_state_hash = Some(state.state_hash.clone());
        Ok(())
    }

    fn calculate_torsion(&self, _previous_hash: &str, _current_hash: &str) -> ResilientResult<f64> {
        Ok(0.1)
    }
}
