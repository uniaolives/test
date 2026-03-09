// arkhe-os/src/bridge/blockchain/ipfs_bridge.rs

use ipfs_api_backend_hyper::{IpfsClient, IpfsApi, TryFromUri};
use crate::orb::core::OrbPayload;
use crate::bridge::BridgeError;
use futures::StreamExt;
use std::io::Cursor;

pub struct IpfsBridge {
    client: IpfsClient,
}

impl IpfsBridge {
    pub fn new() -> Self {
        Self {
            client: IpfsClient::default(),
        }
    }

    /// Publica Orb no IPFS
    pub async fn publish(&self, orb: &OrbPayload) -> Result<String, BridgeError> {
        let data = orb.to_bytes();
        let cursor = Cursor::new(data);

        // Adicionar ao IPFS
        let response = self.client.add(cursor).await
            .map_err(|e| BridgeError::Blockchain(e.to_string()))?;

        // Pin para garantir persistência
        self.client.pin_add(&response.hash, true).await
            .map_err(|e| BridgeError::Blockchain(e.to_string()))?;

        Ok(response.hash)
    }

    /// Recupera Orb do IPFS
    pub async fn retrieve(&self, cid: &str) -> Result<OrbPayload, BridgeError> {
        let data = self.client.cat(cid)
            .map(|res| res.map_err(|e| BridgeError::Blockchain(e.to_string())))
            .collect::<Vec<_>>()
            .await;

        let mut full_data = Vec::new();
        for chunk in data {
            full_data.extend_from_slice(&chunk?);
        }

        Ok(OrbPayload::from_bytes(&full_data)?)
    }
}
