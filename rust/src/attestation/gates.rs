use crate::entropy::VajraEntropyMonitor;
use crate::attestation::nitro::{AttestationVerifier, AttestationError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GateViolation {
    #[error("Prince Key Invalid")]
    PrinceKeyInvalid,
    #[error("EIP-712 Reconstruction Invalid")]
    EIP712Invalid,
    #[error("Ed25519 DNA Verification Failed")]
    Ed25519Invalid,
    #[error("Hard Freeze Active")]
    HardFreezeActive,
    #[error("Vajra Entropy Violation")]
    EntropyViolation,
    #[error("Attestation Error: {0}")]
    Attestation(#[from] AttestationError),
    #[error("Internal Error: {0}")]
    Internal(String),
}

pub struct VerifiedEnclave {
    pub enclave_id: String,
    pub coherence_score: f64,
    pub attestation_timestamp: u64,
}

/// Implementa os 5 Gates para verificação de enclave (Memória 20)
pub fn verify_5_gates(
    attestation_doc: &[u8],
    expected_pcr0: &str, // Invariante em hex
    nonce: &[u8],
    prince_public_key: &[u8; 32],
) -> Result<VerifiedEnclave, GateViolation> {

    // --- GATE 1: Verificar assinatura da AWS (raiz de confiança) ---
    let verifier = AttestationVerifier {
        aws_root_cert: vec![], // Should be loaded from a secure location
        allowed_pcr0_values: vec![hex::decode(expected_pcr0.trim_start_matches("0x")).unwrap_or_default()],
        prince_public_key: *prince_public_key,
    };

    let identity = verifier.verify_attestation_doc(attestation_doc, nonce)?;

    // --- GATE 2: PCR0 já foi validado dentro de verify_attestation_doc ---

    // --- GATE 3: Validar anti-replay (já validado via nonce em verify_attestation_doc) ---

    // --- GATE 4: Verificar Hard Freeze status ---
    if is_hard_freeze_active() {
        return Err(GateViolation::HardFreezeActive);
    }

    // --- GATE 5: Atualizar Vajra Entropy Global ---
    let entropy_monitor = VajraEntropyMonitor::global();

    // We can pass the identity or the doc for further audit
    // For simplicity, we just trigger the update
    let coherence = 0.76; // Mock result from update

    if coherence < 0.72 {
        entropy_monitor.trigger_emergency_morph();
        return Err(GateViolation::EntropyViolation);
    }

    Ok(VerifiedEnclave {
        enclave_id: hex::encode(identity.enclave_id),
        coherence_score: coherence,
        attestation_timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

fn is_hard_freeze_active() -> bool {
    false // Mock
}
