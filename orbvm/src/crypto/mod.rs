pub mod signature;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CryptoConfig {
    pub enabled: bool,
}
