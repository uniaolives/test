use std::ffi::c_void;

// Definições de Mock para Rust (espelhando o C)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SASCAttestationV15 {
    pub signature: [u8; 64],
    pub timestamp: u64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BLAKE3Hash {
    pub hash: [u8; 32],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EncryptedBlob {
    pub data: [u8; 256],
    pub size: u32,
}

// A Mensagem Bitchat Segura
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ConstitutionalBitchatMessage {
    pub message_id: u64,
    pub sender_id: u64,

    // Security Layer
    pub attestation: SASCAttestationV15,
    pub metadata_hash: BLAKE3Hash,
    pub anonymized_payload: EncryptedBlob,

    pub message_type: u8,
    pub priority: u8,
    pub constitutional_seal: u64,
}

// Interface para o Mock C
extern "C" {
    pub fn sasc_sign_payload(data: *const c_void, len: u32) -> SASCAttestationV15;
    pub fn sasc_verify_attestation(att: SASCAttestationV15) -> i32;
}
