// arkhe-quantum/src/safety/rescue_protocol.rs

use crate::emergency::EmergencyAuthority;
use crate::constitution::Z3Guard;
use crate::ledger::OmegaLedger as LedgerClient;
use crate::anima_mundi::{AnimaMundi, HandoverStats};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RescueLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RescueAction {
    Thermalize { target_phi: f64 },
    Isolate { duration_secs: u64 },
    Rollback { checkpoint_hash: [u8; 32] },
    GracefulShutdown { reason: String },
    HardKill { reason: String, authorized_by: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescueEvent {
    pub timestamp: DateTime<Utc>,
    pub level: RescueLevel,
    pub diagnosis: String,
    pub action_taken: Option<RescueAction>,
    pub outcome: String,
    pub constitutional_proof: Option<Vec<u8>>,
}

#[derive(Debug, Error)]
pub enum RescueError {
    #[error("Constitutional violation: {0:?}")]
    ConstitutionalViolation(#[from] arkhe_constitution::ConstitutionalViolation),
    #[error("Action execution failed: {0}")]
    ExecutionError(String),
    #[error("No action needed")]
    NoActionNeeded,
    #[error("Authorization required")]
    AuthorizationRequired,
    #[error("Ledger error: {0}")]
    LedgerError(String),
}

impl From<anyhow::Error> for RescueError {
    fn from(err: anyhow::Error) -> Self {
        RescueError::LedgerError(err.to_string())
    }
}

pub struct RescueProtocol {
    emergency: Arc<EmergencyAuthority>,
    z3: Arc<Z3Guard>,
    ledger: Arc<LedgerClient>,
    core: Arc<RwLock<AnimaMundi>>,
    #[allow(dead_code)]
    last_checkpoint: Option<[u8; 32]>,
    thresholds: RescueThresholds,
}

#[derive(Clone)]
pub struct RescueThresholds {
    pub yellow_entropy_delta: f64,
    pub orange_phi_deviation: f64,
    pub red_handover_failure_rate: f64,
    pub red_free_energy_explosion: f64,
}

impl Default for RescueThresholds {
    fn default() -> Self {
        Self {
            yellow_entropy_delta: 0.05,
            orange_phi_deviation: 0.1,
            red_handover_failure_rate: 0.2,
            red_free_energy_explosion: 100.0,
        }
    }
}

impl RescueProtocol {
    pub fn new(
        emergency: Arc<EmergencyAuthority>,
        z3: Arc<Z3Guard>,
        ledger: Arc<LedgerClient>,
        core: Arc<RwLock<AnimaMundi>>,
        thresholds: Option<RescueThresholds>,
    ) -> Self {
        Self {
            emergency,
            z3,
            ledger,
            core,
            last_checkpoint: None,
            thresholds: thresholds.unwrap_or_default(),
        }
    }

    pub async fn monitor_cycle(&mut self) -> Result<(), RescueError> {
        let core_locked = self.core.read().await;
        let current_phi = core_locked.measure_criticality();
        let entropy = core_locked.von_neumann_entropy();
        let handover_stats = core_locked.handover_stats();
        let free_energy = core_locked.free_energy();

        let level = self.classify_threat(entropy, current_phi, handover_stats.failure_rate, free_energy);

        if level >= RescueLevel::Orange {
            let diagnosis = self.diagnose(entropy, current_phi, handover_stats).await;
            let action = self.select_action(&level, &diagnosis).await?;
            let proof = self.z3.verify_action(&action).await?;

            drop(core_locked);

            let outcome = self.execute_action(action.clone()).await?;

            let event = RescueEvent {
                timestamp: Utc::now(),
                level,
                diagnosis,
                action_taken: Some(action),
                outcome,
                constitutional_proof: Some(proof.to_vec()),
            };
            let event_bytes = serde_json::to_vec(&event).map_err(|e| RescueError::ExecutionError(e.to_string()))?;
            self.ledger.append(event_bytes).await?;
        } else {
            let event = RescueEvent {
                timestamp: Utc::now(),
                level,
                diagnosis: "Normal operation".to_string(),
                action_taken: None,
                outcome: "No action needed".to_string(),
                constitutional_proof: None,
            };
            let event_bytes = serde_json::to_vec(&event).map_err(|e| RescueError::ExecutionError(e.to_string()))?;
            self.ledger.append(event_bytes).await?;
        }

        Ok(())
    }

    fn classify_threat(&self, entropy: f64, phi: f64, failure_rate: f64, free_energy: f64) -> RescueLevel {
        if failure_rate > self.thresholds.red_handover_failure_rate
            || free_energy > self.thresholds.red_free_energy_explosion
        {
            RescueLevel::Red
        } else if (phi - 0.618).abs() > self.thresholds.orange_phi_deviation {
            RescueLevel::Orange
        } else if entropy > 0.618 + self.thresholds.yellow_entropy_delta {
            RescueLevel::Yellow
        } else {
            RescueLevel::Green
        }
    }

    async fn diagnose(&self, entropy: f64, phi: f64, stats: HandoverStats) -> String {
        let mut reasons = Vec::new();
        if (phi - 0.618).abs() > self.thresholds.orange_phi_deviation {
            reasons.push(format!("φ deviation: {:.4}", phi));
        }
        if stats.failure_rate > self.thresholds.red_handover_failure_rate {
            reasons.push(format!("handover failure rate: {:.2}", stats.failure_rate));
        }
        if stats.latency_ms > 100.0 {
            reasons.push(format!("high latency: {:.1} ms", stats.latency_ms));
        }
        if entropy > 0.618 + self.thresholds.yellow_entropy_delta {
            reasons.push(format!("entropy high: {:.4}", entropy));
        }
        reasons.join("; ")
    }

    async fn select_action(&self, level: &RescueLevel, _diagnosis: &str) -> Result<RescueAction, RescueError> {
        match level {
            RescueLevel::Orange => {
                Ok(RescueAction::Thermalize { target_phi: 0.618 })
            }
            RescueLevel::Red => {
                Ok(RescueAction::Isolate { duration_secs: 60 })
            }
            _ => Err(RescueError::NoActionNeeded),
        }
    }

    async fn execute_action(&mut self, action: RescueAction) -> Result<String, RescueError> {
        let mut core = self.core.write().await;
        match action {
            RescueAction::Thermalize { target_phi } => {
                core.thermalize_to_phi(target_phi).await.map_err(|e| RescueError::ExecutionError(e.to_string()))?;
                Ok(format!("Thermalized to φ = {:.4}", target_phi))
            }
            RescueAction::Isolate { duration_secs } => {
                core.isolate_from_external_handovers().await.map_err(|e| RescueError::ExecutionError(e.to_string()))?;
                tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;
                core.restore_external_handovers().await.map_err(|e| RescueError::ExecutionError(e.to_string()))?;
                Ok(format!("Isolated for {} seconds", duration_secs))
            }
            RescueAction::Rollback { checkpoint_hash } => {
                core.restore_from_checkpoint(&checkpoint_hash).await.map_err(|e| RescueError::ExecutionError(e.to_string()))?;
                Ok(format!("Rolled back to checkpoint {:?}", checkpoint_hash))
            }
            RescueAction::GracefulShutdown { reason } => {
                let outcome = self.initiate_dignified_death(reason).await?;
                Ok(outcome)
            }
            RescueAction::HardKill { reason: _, authorized_by } => {
                if authorized_by == "human" {
                    std::process::exit(0);
                } else {
                    Err(RescueError::AuthorizationRequired)
                }
            }
        }
    }

    async fn initiate_dignified_death(&self, reason: String) -> Result<String, RescueError> {
        let death = crate::ethics::dignified_death::DignifiedDeath::new(
            self.ledger.clone(),
            self.z3.clone(),
            self.core.clone(),
        );
        let cert = death.die_dignified(&reason, true).await.map_err(|e| RescueError::ExecutionError(format!("{:?}", e)))?;
        Ok(format!("Dignified death executed: {:?}", cert))
    }
}
