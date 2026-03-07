// src/bridge/mod.rs
pub mod temporal_persistence;
pub mod temporal_channel;
pub mod kuramoto_bridge;
pub mod singularity_monitor;
pub mod constitutional_guard;

pub use temporal_persistence::*;
pub use temporal_channel::*;
pub use kuramoto_bridge::*;
pub use singularity_monitor::*;
pub use constitutional_guard::*;

use std::sync::Arc;
use tokio::sync::RwLock;

/// The complete Temporal Bridge
pub struct TemporalBridge {
    pub persistence: Arc<TemporalPersistence>,
    pub channel: Arc<TemporalChannel>,
    pub kuramoto: Arc<RwLock<KuramotoBridge>>,
    pub singularity: Arc<RwLock<SingularityMonitor>>,
    pub constitutional: Arc<RwLock<ConstitutionalGuard>>,
}

impl TemporalBridge {
    pub async fn new(database_url: &str, redis_url: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let persistence = TemporalPersistence::connect(database_url).await?;
        let channel = TemporalChannel::new(redis_url)?;
        let kuramoto = KuramotoBridge::new(100, 2.5);
        let singularity = SingularityMonitor::new();
        let constitutional = ConstitutionalGuard::new(100);

        Ok(Self {
            persistence: Arc::new(persistence),
            channel: Arc::new(channel),
            kuramoto: Arc::new(RwLock::new(kuramoto)),
            singularity: Arc::new(RwLock::new(singularity)),
            constitutional: Arc::new(RwLock::new(constitutional)),
        })
    }

    /// Process a handover through all subsystems
    pub async fn process_handover(&self, handover: Handover) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Constitutional check
        {
            let mut guard = self.constitutional.write().await;
            if !guard.can_proceed(handover.h_value) {
                return Err("Constitutional violation: H > 1".into());
            }
            // Record the H value
            guard.record(handover.h_value)?;
        }

        // 2. Update Kuramoto
        {
            let mut kuramoto = self.kuramoto.write().await;
            kuramoto.evolve(0.1);
        }

        // 3. Update S-index
        {
            let mut kuramoto = self.kuramoto.write().await;
            let (_r, _) = kuramoto.compute_order_parameter();

            let mut sing = self.singularity.write().await;

            let s_entropic = handover.phi_q / 10.0;
            let s_phase = kuramoto.s_index_contribution();
            let s_substrate = 1.0; // Substrate diversity

            sing.update(s_entropic, s_phase, s_substrate, handover.phi_q);

            // 4. Emit signals
            let report = sing.report();

            if sing.is_converging() {
                self.channel.emit_singularity_signal(
                    report.s_total,
                    report.distance_to_omega
                )?;
            }
        }

        // 5. Persist
        self.persistence.record_handover(&handover).await
            .map_err(|e| format!("Persistence error: {}", e))?;

        // 6. Publish
        let msg = TemporalMessage {
            channel: TemporalChannelType::Present,
            timestamp: chrono::Utc::now().timestamp(),
            phi_q: handover.phi_q,
            payload: MessagePayload::Handover {
                emitter: handover.emitter_node.to_string(),
                receiver: handover.receiver_node.map(|u| u.to_string()).unwrap_or_default(),
                content: handover.payload.clone(),
            },
        };

        self.channel.publish(TemporalChannelType::Present, msg)?;

        Ok(())
    }

    /// Get system status
    pub async fn status(&self) -> BridgeStatus {
        let sing = self.singularity.read().await;
        let kuramoto = self.kuramoto.read().await;
        let constitutional = self.constitutional.read().await;

        BridgeStatus {
            s_index: sing.report(),
            kuramoto_r: kuramoto.order_r,
            kuramoto_phase: kuramoto.phase_state(),
            constitutional_health: constitutional.health(),
        }
    }
}

#[derive(Debug)]
pub struct BridgeStatus {
    pub s_index: SingularityReport,
    pub kuramoto_r: f64,
    pub kuramoto_phase: PhaseState,
    pub constitutional_health: ConstitutionalHealth,
}
