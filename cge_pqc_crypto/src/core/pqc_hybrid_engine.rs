use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn, error, instrument};
use serde::{Serialize, Deserialize};
use crate::algorithms::{PqcAlgorithm, NistSecurityLevel, KemScheme, SignatureScheme, CryptoError, Kyber512Kem, Dilithium3Signature};
use crate::vault::QuantumKeyVault;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HybridMode {
    DoubleSignature {
        pqc_algorithm: PqcAlgorithm,
        classic_algorithm: PqcAlgorithm,
    },
    KemOnly,
    SignatureOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqcConfig {
    pub default_security_level: NistSecurityLevel,
    pub hybrid_mode: HybridMode,
    pub allowed_algorithms: Vec<PqcAlgorithm>,
    pub max_key_size: usize,
    pub quantum_rng: bool,
    pub mandatory_audit: bool,
}

impl Default for PqcConfig {
    fn default() -> Self {
        Self {
            default_security_level: NistSecurityLevel::Level3,
            hybrid_mode: HybridMode::DoubleSignature {
                pqc_algorithm: PqcAlgorithm::Dilithium3,
                classic_algorithm: PqcAlgorithm::Ed25519,
            },
            allowed_algorithms: vec![
                PqcAlgorithm::Kyber768,
                PqcAlgorithm::Dilithium3,
                PqcAlgorithm::Ed25519,
            ],
            max_key_size: 8192,
            quantum_rng: true,
            mandatory_audit: true,
        }
    }
}

pub struct PqcHybridCryptoEngine {
    active_kem: HashMap<NistSecurityLevel, Arc<dyn KemScheme>>,
    active_signature: HashMap<NistSecurityLevel, Arc<dyn SignatureScheme>>,
    pub key_vault: Arc<QuantumKeyVault>,
    config: PqcConfig,
}

impl PqcHybridCryptoEngine {
    #[instrument(name = "pqc_engine_bootstrap", level = "info")]
    pub async fn bootstrap(config: Option<PqcConfig>) -> Result<Arc<Self>, CryptoError> {
        let config = config.unwrap_or_default();
        info!("🔐 Inicializando CGE PQC Hybrid Crypto Engine v31.11-Ω...");

        let key_vault = Arc::new(QuantumKeyVault::new(&config).await?);

        let mut active_kem: HashMap<NistSecurityLevel, Arc<dyn KemScheme>> = HashMap::new();
        active_kem.insert(NistSecurityLevel::Level1, Arc::new(Kyber512Kem));

        let mut active_signature: HashMap<NistSecurityLevel, Arc<dyn SignatureScheme>> = HashMap::new();
        active_signature.insert(NistSecurityLevel::Level2, Arc::new(Dilithium3Signature));

        Ok(Arc::new(Self {
            active_kem,
            active_signature,
            key_vault,
            config,
        }))
    }

    pub fn measure_phi(&self) -> f64 {
        1.038
    }

    pub async fn generate_hybrid_keypair(
        &self,
        key_id: &str,
        security_level: Option<NistSecurityLevel>,
    ) -> Result<HybridKeyPair, CryptoError> {
        let level = security_level.unwrap_or(self.config.default_security_level);

        Ok(HybridKeyPair {
            key_id: key_id.to_string(),
            security_level: level,
            created_at: Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridKeyPair {
    pub key_id: String,
    pub security_level: NistSecurityLevel,
    pub created_at: DateTime<Utc>,
}
