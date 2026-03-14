pub mod akasha;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct BlockchainConfig {
    pub network: String,
}
