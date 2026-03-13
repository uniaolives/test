// crates/arkhe-tor/src/lib.rs
//! Tor transport privacy layer for ArkheNet.
//! Provides high-level management of Tor circuits, onion services, and SOCKS5 proxying.

use std::sync::Arc;
use arti_client::{TorClient, TorClientConfig};
use tor_rtcompat::tokio::TokioNativeTlsRuntime;
use anyhow::{Result, anyhow};
use tracing::{info, error};

pub use arti_client;

/// Manages Tor connectivity and hidden services for a Bio-Node.
pub struct TorManager {
    client: TorClient<TokioNativeTlsRuntime>,
}

impl TorManager {
    /// Bootstraps a new Tor client.
    pub async fn bootstrap() -> Result<Self> {
        info!("🜏 Bootstrapping Tor client (Onionizing the Tzinor)...");

        let config = TorClientConfig::default();
        let runtime = TokioNativeTlsRuntime::current()?;

        let client = TorClient::with_runtime(runtime)
            .config(config)
            .create_bootstrapped()
            .await
            .map_err(|e| anyhow!("Tor bootstrap failed: {}", e))?;

        info!("🜏 Tor client bootstrapped successfully");
        Ok(Self { client })
    }

    /// Returns a reference to the underlying Tor client.
    pub fn client(&self) -> &TorClient<TokioNativeTlsRuntime> {
        &self.client
    }

    /// Establishes a connection to a remote .onion address.
    pub async fn connect(&self, addr: &str, port: u16) -> Result<arti_client::DataStream> {
        info!("🜏 Connecting to {} via Tor circuit...", addr);
        self.client.connect((addr, port)).await
            .map_err(|e| anyhow!("Failed to connect to node via Tor: {}", e))
    }
}
