pub mod nitro;
pub mod gates;

#[derive(Debug, Clone)]
pub struct SASCAttestation {
    pub signature: String,
}

#[derive(Debug, Clone)]
pub struct PrinceSignature {
    pub value: String,
}

impl PrinceSignature {
    pub fn sign_genesis(_phase: &crate::clock::SchumannPhase, _stability: &crate::entropy::PhiStabilityProof) -> Result<Self, &'static str> {
        Ok(Self { value: "prince_sig_alpha_1".to_string() })
    }
}
