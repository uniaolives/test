// pleroma_quantum mock stub
pub struct SharedEntanglement;
impl SharedEntanglement {
    pub fn coherence(&self) -> f64 { 0.98 }
    pub fn nonce(&self) -> u64 { 12345 }
}

pub fn derive_key(coherence: f64, nonce: u64) -> Vec<u8> {
    vec![0u8; 32]
}
