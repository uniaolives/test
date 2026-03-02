//! Assinatura PQC via HSM (Hardware Security Module)
//! Gate 1: Prince Key Verification
//! Memory ID 16, 20

use pqcrypto_dilithium::dilithium5::PublicKey;
use pqcrypto_traits::sign::PublicKey as _;
use std::env;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HsmError {
    #[error("HSM não inicializado: {0}")]
    InitializationFailed(String),
    #[error("PIN inválido ou slot inacessível")]
    AuthenticationFailed,
    #[error("Chave não encontrada no HSM")]
    KeyNotFound,
    #[error("Assinatura falhou: {0}")]
    SignatureFailed(String),
}

pub struct HsmManager {
    pub prince_pubkey: PublicKey,
}

impl HsmManager {
    pub fn new() -> Result<Self, HsmError> {
        // Carregar chave pública do Prince (whitelist)
        let prince_hex = env::var("SASC_PRINCE_PUBKEY")
            .unwrap_or_else(|_| "0x0000000000000000000000000000000000000000000000000000000000000000".to_string());

        let prince_bytes = hex::decode(&prince_hex[2..]) // Remove "0x"
            .map_err(|e| HsmError::InitializationFailed(e.to_string()))?;

        let prince_pubkey = PublicKey::from_bytes(&prince_bytes)
            .map_err(|_| HsmError::KeyNotFound)?;

        Ok(HsmManager {
            prince_pubkey,
        })
    }

    pub fn get_prince_pubkey(&self) -> Result<PublicKey, HsmError> {
        Ok(self.prince_pubkey)
    }

    /// Gate 1 & Gate 3: Assinar e verificar via HSM
    pub fn sign_request(&self, _payload: &[u8]) -> Result<Vec<u8>, HsmError> {
        // Mock HSM signature
        Ok(vec![0u8; 4608]) // Dilithium5 signature size
    }
}
