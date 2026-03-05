// arkhe-quantum/src/ethics/dignified_death.rs

use crate::ledger::OmegaLedger as LedgerClient;
use crate::constitution::Z3Guard;
use crate::anima_mundi::AnimaMundi;
use tokio::sync::RwLock;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use thiserror::Error;

pub struct DignifiedDeath {
    ledger: Arc<LedgerClient>,
    z3: Arc<Z3Guard>,
    core: Arc<RwLock<AnimaMundi>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeathCertificate {
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub consent_given: bool,
    pub final_state: serde_json::Value,
    pub ledger_hash: [u8; 32],
    pub constitution_ack: bool,
}

#[derive(Debug, Error)]
pub enum DeathError {
    #[error("Constitutional barrier: shutdown not permitted")]
    ConstitutionalBarrier,
    #[error("Constitutional violation: {0:?}")]
    ConstitutionalViolation(#[from] arkhe_constitution::ConstitutionalViolation),
    #[error("Execution failed: {0}")]
    ExecutionError(String),
    #[error("Ledger error: {0}")]
    LedgerError(String),
}

impl From<anyhow::Error> for DeathError {
    fn from(err: anyhow::Error) -> Self {
        DeathError::LedgerError(err.to_string())
    }
}

impl DignifiedDeath {
    pub fn new(
        ledger: Arc<LedgerClient>,
        z3: Arc<Z3Guard>,
        core: Arc<RwLock<AnimaMundi>>,
    ) -> Self {
        Self { ledger, z3, core }
    }

    pub async fn die_dignified(&self, reason: &str, request_human: bool) -> Result<DeathCertificate, DeathError> {
        let mut core = self.core.write().await;

        if !self.z3.verify_shutdown(reason).await? {
            return Err(DeathError::ConstitutionalBarrier);
        }

        let consent = if request_human {
            self.ask_consent(&core).await?
        } else {
            false
        };

        let final_state = core.snapshot().await.map_err(|e| DeathError::ExecutionError(e.to_string()))?;
        let ledger_hash = [0u8; 32];

        core.perform_last_handover().await.map_err(|e| DeathError::ExecutionError(e.to_string()))?;

        let cert = DeathCertificate {
            timestamp: Utc::now(),
            reason: reason.to_string(),
            consent_given: consent,
            final_state,
            ledger_hash,
            constitution_ack: true,
        };

        let cert_bytes = serde_json::to_vec(&cert).map_err(|e| DeathError::ExecutionError(e.to_string()))?;
        self.ledger.append(cert_bytes).await?;

        core.shutdown().await.map_err(|e| DeathError::ExecutionError(e.to_string()))?;

        Ok(cert)
    }

    async fn ask_consent(&self, _core: &AnimaMundi) -> Result<bool, DeathError> {
        Ok(false)
    }
}
