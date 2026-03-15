// arkhe-os/src/net/tor_layer.rs
//! Tor transport-layer privacy for ArkheOS.
//! Provides unified management of .onion hidden services and censorship-resistant routing.

use std::sync::Arc;
use arkhe_tor::TorManager;
use anyhow::Result;
use tracing::info;

/// High-level interface for 'Onionizing' the Arkhe stack.
pub struct ArkheTorLayer {
    manager: Arc<TorManager>,
}

impl ArkheTorLayer {
    /// Bootstraps the Tor layer for a Bio-Node.
    pub async fn bootstrap() -> Result<Self> {
        let manager = TorManager::bootstrap().await?;
        Ok(Self {
            manager: Arc::new(manager),
        })
    }

    /// Connects to a remote agent via its hidden service identity.
    pub async fn connect_to_agent(&self, onion_addr: &str, port: u16) -> Result<arti_client::DataStream> {
        self.manager.connect(onion_addr, port).await
    }

    /// Returns the Tor manager for lower-level access.
    pub fn manager(&self) -> Arc<TorManager> {
        self.manager.clone()
    }
}
