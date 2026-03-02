// rust/src/wallet/mod.rs
pub mod manager;
pub mod keychain;

pub use manager::WalletManager;
pub use keychain::{Keychain, KeyType};

// Estruturas públicas
#[derive(Debug, Clone)]
pub struct WalletInfo {
    pub address: String,
    pub balance: u64,
    pub network: Network,
    pub last_sync: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Network {
    Mainnet,
    Testnet,
    Turbo,  // Rede de teste gratuita
}
