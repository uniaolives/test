pub mod client;
pub mod mapper;
pub mod sync;
pub mod types;

use arkhe_manifold::GlobalManifold;
use std::time::Duration;

pub struct BridgeConfig {
    pub poll_interval_ms: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self { poll_interval_ms: 1000 }
    }
}

pub struct FoundryBridge {
    pub manifold_cache: GlobalManifold,
    pub config: BridgeConfig,
    pub last_sync: i64,
}

impl FoundryBridge {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            manifold_cache: GlobalManifold::new(),
            config: BridgeConfig::default(),
            last_sync: 0,
        })
    }

    pub async fn run(&mut self) -> ! {
        let mut interval = tokio::time::interval(Duration::from_millis(self.config.poll_interval_ms));

        loop {
            interval.tick().await;
            // Simulated Polling iteration
        }
    }
}
