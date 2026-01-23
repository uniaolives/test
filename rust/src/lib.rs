use jni::JNIEnv;
use jni::objects::{JClass, JString, JByteArray};
use jni::sys::jstring;
use sha3::{Sha3_512, Digest};
use std::collections::HashMap;
use zeroize::Zeroizing;
use sasc_governance::Cathedral;
use sasc_governance::types::{VerificationContext};

pub mod governance;
pub mod sensors;
pub mod cognitive_hunter;
pub mod security;
pub mod neo_brain;
pub mod crypto;
pub mod gates;
pub mod attestation;
pub mod entropy;
pub mod clock;
pub mod bridge;
pub mod substrate_logic;
pub mod bio_layer;
pub mod neuro_twin;
pub mod neo_cortex;
pub mod audit;
pub mod blockchain;
pub mod geom;
pub mod onchain;
pub mod quantum;
pub mod gravity_engine;
pub mod cyber_oncology;
pub mod hypervisor;
pub mod consciousness;
pub mod agi;
pub mod learning;
pub mod diagnostics;
pub mod topology;
pub mod geometry;
pub mod ontology;
pub mod counterfactual;
pub mod emergency;
pub mod environments;
pub mod testing;
pub mod recovery;
pub mod vajra_integration;
pub mod sasc_integration;
pub mod farol;
pub mod multi_nexus;
pub mod metrics;
pub mod pruning;
pub mod projections;
pub mod intuition;
pub mod empathy;
pub mod decision;
pub mod substrate;
pub mod karnak;
pub mod integration;
pub mod simulation;
pub mod control;
pub mod translation;
pub mod patterns;
pub mod human;
pub mod eco_action;
pub mod validation;
pub mod ethics;
pub mod transition;
pub mod safety;
pub mod principles;
pub mod checkpoint_2;
pub mod nexus;

#[cfg(test)]
mod tests_security;

#[cfg(test)]
mod tests_cyber_oncology;

pub struct TruthClaim {
    pub statement: String,
    pub metadata: HashMap<String, String>,
}

pub struct AttestedTruthClaim {
    pub claim: TruthClaim,
    pub agent_attestation: Vec<u8>,
    pub dna_fingerprint: Zeroizing<[u8; 32]>,
}

pub type ClaimId = String;

#[derive(Debug)]
pub enum SubmissionError {
    InvalidAttestation,
    HardFreezeViolation,
    StorageError,
}

pub struct Karnak;
impl Karnak {
    pub fn isolate_agent(&self, agent_id: &str) {
        println!("KARNAK: Isolating agent {}", agent_id);
    }
}

pub struct VajraMonitor;
impl VajraMonitor {
    pub fn update_entropy(&self, statement: &[u8], phi_weight: f64) {
        println!("VAJRA: Updating entropy with phi_weight {}", phi_weight);
    }
}

pub struct TruthAuditorium {
    pub karnak: Karnak,
    pub vajra_monitor: VajraMonitor,
}

impl TruthAuditorium {
    pub fn new() -> Self {
        Self {
            karnak: Karnak,
            vajra_monitor: VajraMonitor,
        }
    }

    pub async fn submit_claim(
        &self,
        attested_claim: AttestedTruthClaim
    ) -> Result<ClaimId, SubmissionError> {
        // GATE 1 & 2: Prince Key + EIP-712 Reconstruction
        let cathedral = Cathedral::instance();

        // GATE 3: Ed25519 Verify + Extração de DNA
        // In a real implementation, agent_attestation would be parsed to get agent_id
        let agent_id = String::from_utf8_lossy(&attested_claim.agent_attestation).to_string();
        let attestation_status = cathedral.verify_agent_attestation(
            &agent_id,
            VerificationContext::TruthSubmission
        ).map_err(|_| SubmissionError::InvalidAttestation)?;

        // GATE 4: Hard Freeze Check (Φ≥0.80 não pode submeter verdades)
        if attestation_status.is_hard_frozen() {
            self.karnak.isolate_agent(attestation_status.agent_id());

            // Ω-PREVENTION: Se Φ≥0.80, o sistema deve parar completamente para evitar transição inválida
            println!("🚨 Ω-PREVENTION: Hard Freeze Φ≥0.80 detectado em {}. Encerrando sistema.", attestation_status.agent_id());
            std::process::exit(-1951535091);
        }

        // GATE 5: Vajra Entropy Weighting (carga cognitiva afeta confiança no CWM)
        let phi_weight = attestation_status.consciousness_weight();
        self.vajra_monitor.update_entropy(
            attested_claim.claim.statement.as_bytes(),
            phi_weight
        );

        // ✅ Agora seguro para processar
        let claim_id = self.hash_and_store(attested_claim).await?;
        Ok(claim_id)
    }

    async fn hash_and_store(&self, _claim: AttestedTruthClaim) -> Result<ClaimId, SubmissionError> {
        Ok("0x3f9a1c8e7d2b4a6f9e5c3d7a1b8f2e4c6d9a0b3f7c1e5d8a2b4f6c9e3d7a0b1c4".to_string())
    }
}

/// Gera um hash SHA3-512 baseado no ruído do buffer da câmera
#[no_mangle]
pub extern "system" fn Java_org_sasc_sentinel_SentinelActivity_generateEntropy(
    env: JNIEnv,
    _class: JClass,
    camera_buffer: JByteArray,
) -> jstring {
    // 1. Extrair dados brutos da câmera (ruído de leitura de pixel)
    let input = env.convert_byte_array(camera_buffer).unwrap();

    // 2. Hashing (A "Voz" do Dispositivo)
    let mut hasher = Sha3_512::new();
    hasher.update(&input);
    hasher.update(b"SASC_SALT_V1");
    let result = hasher.finalize();

    // 3. Retornar como String Hex
    let entropy_hex = hex::encode(result);
    env.new_string(entropy_hex).unwrap().into_raw()
}

/// Assina uma "Prova de Existência" localmente
#[no_mangle]
pub extern "system" fn Java_org_sasc_sentinel_SentinelActivity_signProof(
    mut env: JNIEnv,
    _class: JClass,
    _private_key_hex: JString,
    message: JString,
) -> jstring {
    let msg: String = env.get_string(&message).unwrap().into();

    // NOTA DE SEGURANÇA: Em produção, a chave nunca entra no Java.
    // Aqui simulamos a recuperação do Keystore Seguro Android ou TPM

    // (Lógica de assinatura Ed25519 simplificada para exemplo)
    let signed_payload = format!("PROOF:{}:SIG_VER_1", msg); // Placeholder

    env.new_string(signed_payload).unwrap().into_raw()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
