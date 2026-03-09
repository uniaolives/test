// src/bridge/temporal_channel.rs
use redis::{Client, Commands};
use serde::{Serialize, Deserialize};
use tokio::sync::broadcast;

/// Channels for temporal communication
pub struct TemporalChannel {
    client: Client,
    /// Internal broadcast for local subscribers
    internal: broadcast::Sender<TemporalMessage>,
}

/// Message types for temporal layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMessage {
    pub channel: TemporalChannelType,
    pub timestamp: i64,
    pub phi_q: f64,
    pub payload: MessagePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalChannelType {
    /// Current layer (2026)
    Present,
    /// Ancestral layer (2008)
    Ancestral,
    /// Omega layer (2140)
    Omega,
    /// Constitutional alerts
    Constitutional,
    /// Singularity tracking
    Singularity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    Handover { emitter: String, receiver: String, content: String },
    PhaseLock { kuramoto_r: f64 },
    ConstitutionalAlert { h_value: f64 },
    SingularityApproach { s_index: f64, distance_to_omega: f64 },
    GhostCluster { orbit_id: String, stability: f64 },
}

impl TemporalChannel {
    pub fn new(redis_url: &str) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;
        let (tx, _) = broadcast::channel(1024);

        Ok(Self {
            client,
            internal: tx,
        })
    }

    /// Publish to a temporal layer
    pub fn publish(&self, channel: TemporalChannelType, message: TemporalMessage) -> Result<(), redis::RedisError> {
        let mut conn = self.client.get_connection()?;
        let channel_name = match channel {
            TemporalChannelType::Present => "arkhe:2026",
            TemporalChannelType::Ancestral => "arkhe:2008",
            TemporalChannelType::Omega => "arkhe:2140",
            TemporalChannelType::Constitutional => "arkhe:constitutional",
            TemporalChannelType::Singularity => "arkhe:singularity",
        };

        let payload = serde_json::to_string(&message).unwrap();
        let _: () = conn.publish(channel_name, payload)?;

        // Also broadcast locally
        let _ = self.internal.send(message);

        Ok(())
    }

    /// Subscribe to a temporal layer
    pub fn subscribe(&self, _channel: TemporalChannelType) -> broadcast::Receiver<TemporalMessage> {
        self.internal.subscribe()
    }

    /// Emit constitutional alert if H > 1.0
    pub fn emit_constitutional_warning(&self, h_value: f64) -> Result<(), redis::RedisError> {
        let message = TemporalMessage {
            channel: TemporalChannelType::Constitutional,
            timestamp: chrono::Utc::now().timestamp(),
            phi_q: 0.0,
            payload: MessagePayload::ConstitutionalAlert { h_value },
        };

        self.publish(TemporalChannelType::Constitutional, message)
    }

    /// Emit singularity approach signal
    pub fn emit_singularity_signal(&self, s_index: f64, distance: f64) -> Result<(), redis::RedisError> {
        let message = TemporalMessage {
            channel: TemporalChannelType::Singularity,
            timestamp: chrono::Utc::now().timestamp(),
            phi_q: 0.0,
            payload: MessagePayload::SingularityApproach {
                s_index,
                distance_to_omega: distance,
            },
        };

        self.publish(TemporalChannelType::Singularity, message)
    }
}
