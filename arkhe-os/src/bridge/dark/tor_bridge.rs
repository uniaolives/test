// arkhe-os/src/bridge/dark/tor_bridge.rs

use arti_client::{TorClient, TorAddr, DataStream};
use crate::propagation::payload::OrbPayload;
use crate::bridge::BridgeError;
use std::str::FromStr;
use tokio::io::AsyncWriteExt;
use tor_rtcompat::Runtime;

pub struct TorBridge<R: Runtime> {
    client: TorClient<R>,
    hidden_services: Vec<String>,
}

impl<R: Runtime> TorBridge<R> {
    pub fn new(client: TorClient<R>, services: Vec<String>) -> Self {
        Self { client, hidden_services: services }
    }

    /// Envia Orb via Tor hidden service
    pub async fn send(&self, orb: &OrbPayload) -> Result<(), BridgeError> {
        let data = orb.to_bytes();

        for service in &self.hidden_services {
            let addr = TorAddr::from_str(service)
                .map_err(|e| BridgeError::Tor(e.to_string()))?;

            let mut stream: DataStream = self.client.connect(addr)
                .await
                .map_err(|e: arti_client::Error| BridgeError::Tor(e.to_string()))?;

            // Enviar Orb
            stream.write_all(&data).await
                .map_err(|e: std::io::Error| BridgeError::Tor(e.to_string()))?;
        }

        Ok(())
    }
}
