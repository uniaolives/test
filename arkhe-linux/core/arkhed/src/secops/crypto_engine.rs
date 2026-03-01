use std::collections::HashMap;
use arkhe_quantum::Handover;
use pqcrypto_kyber::kyber1024 as kyber;
use pqcrypto_traits::sign::PublicKey as _;
use pqcrypto_traits::kem::PublicKey as _;

pub struct CryptoEngine {
    key_store: HashMap<[u8; 8], Vec<u8>>,
    entropy_threshold: f64,
}

impl CryptoEngine {
    pub fn new() -> Self {
        Self {
            key_store: HashMap::new(),
            entropy_threshold: 0.3,
        }
    }

    pub fn verify_handover(&self, h: &Handover) -> bool {
        // Ω+207: Utiliza a lógica de verificação Dilithium5 implementada em arkhe-quantum
        if let Some(pk_bytes) = self.key_store.get(&h.header.emitter_id.to_le_bytes()) {
            h.verify(pk_bytes)
        } else {
            !h.signature.is_empty()
        }
    }

    pub fn monitor_channel_entropy(&mut self, node_a: [u8; 8], node_b: [u8; 8], entropy: f64) {
        if entropy < self.entropy_threshold {
            self.rotate_keys(node_a, node_b);
        }
    }

    fn rotate_keys(&mut self, node_a: [u8; 8], node_b: [u8; 8]) {
        let (pk, _sk) = kyber::keypair();
        self.key_store.insert(node_a, pqcrypto_traits::kem::PublicKey::as_bytes(&pk).to_vec());
        tracing::info!("Rotating Kyber1024 keys for channel {}-{}", hex::encode(node_a), hex::encode(node_b));
    }
}
