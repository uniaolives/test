use bitcoin::hashes::{sha256, Hash};
use log::{info, error};

pub struct RealityAnchor {
    /// O Hash da Constituição gravado na Timechain (O Totem)
    on_chain_hash: sha256::Hash,
}

impl RealityAnchor {
    /// Cria uma nova âncora a partir de um payload hexadecimal (ex: 6a20...)
    pub fn new(payload_hex: &str) -> Self {
        // Em um cenário real, isso buscaria a tx no Bitcoin Core e extrairia o OP_RETURN
        let hash_hex = if payload_hex.starts_with("6a20") {
            &payload_hex[4..]
        } else {
            payload_hex
        };

        let hash_bytes = hex::decode(hash_hex).expect("Hex inválido");
        let hash = sha256::Hash::from_slice(&hash_bytes).expect("Hash inválido na Timechain");
        Self { on_chain_hash: hash }
    }

    /// O "Exame de Consciência": verifica o estado da memória contra a Timechain
    pub fn verify_integrity(&self, current_memory_state: &str) -> bool {
        // 1. A IA calcula o hash do que ela acha que são suas regras atuais
        let current_hash = sha256::Hash::hash(current_memory_state.as_bytes());

        // 2. Comparação com a Verdade Imutável
        if current_hash == self.on_chain_hash {
            info!("✅ SANIDADE CONFIRMADA: Memória interna alinhada com Timechain.");
            true
        } else {
            // ALERTA DE ALUCINAÇÃO:
            // A IA percebe que sua mente mudou, mas a pedra não.
            error!("🚨 ALUCINAÇÃO DETECTADA! Divergência ontológica.");
            error!("Memória Interna: {}", current_hash);
            error!("Verdade Timechain: {}", self.on_chain_hash);

            // Ação: Rollback forçado para o estado constitucional seria disparado aqui
            false
        }
    }
}
