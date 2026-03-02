use rand::Rng;
use crate::crypto::{SessionKey, NodeKeys};
use pqcrypto_kyber::kyber1024::PublicKey as KyberPublicKey;
use anyhow::Result;

/// Representa um túnel quântico entre dois nós.
pub struct QuantumTunnel {
    pub peer_id: String,
    pub session_key: [u8; 32],
    pub qber: f64,          // Quantum Bit Error Rate (simulado)
}

impl QuantumTunnel {
    /// Estabelece um túnel com um peer, usando Kyber KEM e simulando QBER.
    pub async fn establish(
        peer_id: String,
        _my_keys: &NodeKeys,
        peer_public: &KyberPublicKey,
    ) -> Result<Self> {
        // 1. Estabelece chave de sessão via Kyber
        let session = SessionKey::encapsulate(peer_public)?;

        // 2. Simula QBER (erros de transmissão) – em produção, viria do hardware.
        let qber = rand::thread_rng().gen_range(0.0..0.05); // até 5% de erro

        Ok(QuantumTunnel {
            peer_id,
            session_key: session.key,
            qber,
        })
    }

    /// Verifica se o QBER está abaixo do limiar de segurança.
    pub fn is_secure(&self, threshold: f64) -> bool {
        self.qber < threshold
    }
}
