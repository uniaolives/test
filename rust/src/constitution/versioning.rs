// rust/src/constitution/versioning.rs
use crate::error::{ResilientError, ResilientResult};

pub struct ConstitutionVersion {
    pub version: String,
}

pub struct VersionManager {
    current_version: String,
}

impl VersionManager {
    pub fn new(version: String) -> Self {
        Self { current_version: version }
    }

    pub fn validate_version(&self, version: &str) -> ResilientResult<()> {
        if version != self.current_version {
            return Err(ResilientError::VersionMismatch {
                expected: self.current_version.clone(),
                actual: version.to_string(),
            });
        }
        Ok(())
    }
}
