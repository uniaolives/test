// rust/src/wallet/manager.rs
use crate::error::{ResilientError, ResilientResult};
use crate::wallet::Network;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// Mock structures to replace arweave_rs if not available
pub struct MockWallet;
impl MockWallet {
    pub fn new() -> Self { Self }
    pub fn from_file(_path: &PathBuf) -> Result<Self, String> { Ok(Self) }
    pub fn to_file(&self, _path: &PathBuf) -> Result<(), String> { Ok(()) }
    pub fn get_address(&self) -> String { "mock-arweave-address".to_string() }
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>, String> { Ok(data.to_vec()) }
}

pub struct MockArweave;
impl MockArweave {
    pub fn new(_url: &str) -> Self { Self }
    pub async fn get_balance(&self, _address: &str) -> Result<u64, String> { Ok(1000) }
}

#[derive(Clone)]
pub struct WalletManager {
    wallet: Arc<RwLock<Option<MockWallet>>>,
    arweave: Arc<MockArweave>,
    #[allow(dead_code)]
    wallet_path: PathBuf,
    network: Network,
    balance_cache: Arc<RwLock<u64>>,
}

impl WalletManager {
    pub async fn new(config_path: Option<PathBuf>) -> ResilientResult<Self> {
        let path = config_path.unwrap_or_else(|| {
            let mut dir = PathBuf::from(".config");
            dir.push("resilient-agent/wallet.json");
            dir
        });

        let wallet = MockWallet::new();

        let network = match std::env::var("ARWEAVE_NETWORK")
            .unwrap_or_else(|_| "turbo".to_string())
            .as_str()
        {
            "mainnet" => Network::Mainnet,
            "testnet" => Network::Testnet,
            _ => Network::Turbo,
        };

        let arweave = match network {
            Network::Mainnet => MockArweave::new("https://arweave.net"),
            Network::Testnet => MockArweave::new("https://testnet.arweave.net"),
            Network::Turbo => MockArweave::new("https://turbo.arweave.net"),
        };

        Ok(Self {
            wallet: Arc::new(RwLock::new(Some(wallet))),
            arweave: Arc::new(arweave),
            wallet_path: path,
            network,
            balance_cache: Arc::new(RwLock::new(0)),
        })
    }

    pub async fn get_address(&self) -> ResilientResult<String> {
        let wallet = self.wallet.read().await;
        match &*wallet {
            Some(w) => Ok(w.get_address()),
            None => Err(ResilientError::Wallet("Wallet not loaded".to_string())),
        }
    }

    pub async fn sync_balance(&self) -> ResilientResult<u64> {
        let address = self.get_address().await?;

        let balance = match self.network {
            Network::Turbo => 0,
            _ => {
                self.arweave.get_balance(&address).await
                    .map_err(|e| ResilientError::Wallet(format!("Balance query failed: {}", e)))?
            }
        };

        *self.balance_cache.write().await = balance;
        Ok(balance)
    }

    pub async fn estimate_storage_cost(&self, data_size: usize) -> ResilientResult<u64> {
        if data_size <= 102_400 && self.network == Network::Turbo {
            return Ok(0);
        }

        let mb = (data_size as f64 / 1_048_576.0).ceil() as u64;
        let cost = mb * 1_000_000_000;

        Ok(cost)
    }

    pub async fn sign_data(&self, data: &[u8]) -> ResilientResult<Vec<u8>> {
        let wallet = self.wallet.read().await;
        match &*wallet {
            Some(w) => {
                w.sign(data)
                    .map_err(|e| ResilientError::Wallet(format!("Signing failed: {}", e)))
            }
            None => Err(ResilientError::Wallet("Wallet not loaded".to_string())),
        }
    }

    pub fn network(&self) -> Network {
        self.network
    }
}
