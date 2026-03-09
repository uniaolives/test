// arkhe-os/src/bridge/dark/tor_bridge.rs

use arkhe_tor::arti_client::{TorClient, TorAddr};
use arti_client::{TorClient, TorAddr};
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;
use std::str::FromStr;
use tokio::io::AsyncWriteExt;

pub struct TorBridge {
    client: TorClient<arti_client::DefaultRuntime>,
    hidden_services: Vec<String>,
}

impl TorBridge {
    pub fn new(client: TorClient<arkhe_tor::arti_client::DefaultRuntime>, services: Vec<String>) -> Self {
    pub fn new(client: TorClient<arti_client::DefaultRuntime>, services: Vec<String>) -> Self {
        Self { client, hidden_services: services }
    }

    /// Envia Orb via Tor hidden service
    pub async fn send(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let data = orb.to_bytes();

        for service in &self.hidden_services {
            let addr = TorAddr::from_str(service)
                .map_err(|e| BridgeError::Tor(e.to_string()))?;
            let mut stream = self.client.connect(addr)
                .await
                .map_err(|e| BridgeError::Tor(e.to_string()))?;

            // Enviar Orb
            stream.write_all(&data).await
                .map_err(|e| BridgeError::Tor(e.to_string()))?;
        }

        Ok(())
    }
}
