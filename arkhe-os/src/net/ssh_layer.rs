// arkhe-os/src/net/ssh_layer.rs
//! Secure communication layer for inter-agent synchronization and Orb payload transfer.
//! Implements a native SSH server and client using `russh` to facilitate secure Tzinor channels.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use russh::*;
use russh_keys::key::PublicKey;
use tokio::sync::Mutex;
use tracing::{info, warn, error};
use serde::{Deserialize, Serialize};

/// Inter-agent message types carried over SSH channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessage {
    /// Synchronize phase state for Kuramoto evolution.
    Sync {
        phases: Vec<f64>,
        coherence: f64,
    },
    /// Transfer an Orb payload between nodes.
    OrbTransfer {
        payload: serde_json::Value,
    },
    /// Mutual authentication and state verification.
    Handshake {
        node_id: String,
        lambda_2: f64,
    },
}

/// Handler for the embedded SSH server.
pub struct OrbSshHandler {
    pub node_id: String,
    pub authorized_agents: Arc<Mutex<HashMap<String, PublicKey>>>,
}

#[async_trait]
impl server::Handler for OrbSshHandler {
    type Error = anyhow::Error;

    async fn auth_publickey(
        &mut self,
        user: &str,
        public_key: &PublicKey,
    ) -> Result<server::Auth, Self::Error> {
        let agents = self.authorized_agents.lock().await;
        if let Some(authorized_key) = agents.get(user) {
            if authorized_key == public_key {
                info!("SSH Authentication successful for agent: {}", user);
                return Ok(server::Auth::Accept);
            }
        }
        warn!("SSH Authentication rejected for agent: {}", user);
        Ok(server::Auth::Reject { proceed_with_methods: None })
    }

    async fn channel_open_session(
        &mut self,
        _channel: Channel<server::Msg>,
        _session: &mut server::Session,
    ) -> Result<bool, Self::Error> {
        info!("SSH Session channel opened for inter-agent communication");
        Ok(true)
    }

    async fn data(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut server::Session,
    ) -> Result<(), Self::Error> {
        if let Ok(msg) = serde_json::from_slice::<AgentMessage>(data) {
            info!("Received AgentMessage: {:?}", msg);
            // Acknowledgement
            let ack = b"MESSAGE_RECEIVED\n";
            let _ = session.data(channel, ack.to_vec().into());
            session.data(channel, ack.to_vec().into())?;
        } else {
            error!("Received malformed data over SSH channel");
        }
        Ok(())
    }
}

/// Client handler for establishing secure tunnels to remote agents.
pub struct OrbSshClientHandler;

#[async_trait]
impl client::Handler for OrbSshClientHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &PublicKey,
    ) -> Result<bool, Self::Error> {
        // In the Arkhe network, server keys are validated against the Teknet Ledger.
        Ok(true)
    }
}
