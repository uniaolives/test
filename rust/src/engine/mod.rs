use crate::civilization::foundry::CivilizationContract;

pub struct NeoEngine;

impl NeoEngine {
    pub async fn mint_genesis(_contract: &CivilizationContract) -> Result<String, &'static str> {
        log::info!("NEO_ENGINE: Minting Genesis Block #15");
        Ok("0xGENESIS-CIV-7f9a2b1c3d4e5f6a".to_string())
    }
}

pub mod neo_engine {
    pub use super::NeoEngine;
    use crate::civilization::foundry::CivilizationContract;

    pub async fn mint_genesis(contract: &CivilizationContract) -> Result<String, &'static str> {
        super::NeoEngine::mint_genesis(contract).await
    }
}
