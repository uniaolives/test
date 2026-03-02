use aws_nitro_enclaves_cose::CoseSign1;
use x509_parser::prelude::*;
use ring::digest;
use std::time::SystemTime;
use thiserror::Error;
use serde::{Deserialize};
use ciborium::from_reader;

#[derive(Debug, Error)]
pub enum AttestationError {
    #[error("COSE error: {0}")]
    CoseError(String),
    #[error("X509 error: {0}")]
    X509Error(String),
    #[error("CBOR error: {0}")]
    CborError(String),
    #[error("Missing user data")]
    MissingUserData,
    #[error("Invalid nonce")]
    InvalidNonce,
    #[error("Root certificate mismatch")]
    RootCertMismatch,
    #[error("Invalid root signature")]
    InvalidRootSignature,
    #[error("Invalid certificate chain")]
    InvalidCertChain,
    #[error("Unauthorized PCR0")]
    UnauthorizedPCR0,
    #[error("Audit log failed")]
    AuditLogFailed,
    #[error("Karnak isolation failed")]
    KarnakIsolationFailed,
}

impl From<aws_nitro_enclaves_cose::error::CoseError> for AttestationError {
    fn from(e: aws_nitro_enclaves_cose::error::CoseError) -> Self {
        AttestationError::CoseError(format!("{:?}", e))
    }
}

pub struct EnclaveIdentity {
    pub enclave_id: digest::Digest,
    pub pcrs: Vec<Vec<u8>>,
    pub attestation_timestamp: SystemTime,
}

/// Represents the structure of an AWS Nitro Enclave attestation document.
/// Based on AWS documentation: https://docs.aws.amazon.com/enclaves/latest/userguide/set-up-attestation.html
#[derive(Deserialize, Debug)]
pub struct NitroAttestationDoc {
    #[serde(rename = "module_id")]
    pub module_id: String,
    pub timestamp: u64,
    pub digest: String,
    pub pcrs: std::collections::HashMap<u32, Vec<u8>>,
    pub certificate: Vec<u8>,
    pub cabundle: Vec<Vec<u8>>,
    #[serde(rename = "public_key")]
    pub public_key: Option<Vec<u8>>,
    #[serde(rename = "user_data")]
    pub user_data: Option<Vec<u8>>,
    #[serde(rename = "nonce")]
    pub nonce: Option<Vec<u8>>,
}

pub struct AttestationVerifier {
    pub aws_root_cert: Vec<u8>,
    pub allowed_pcr0_values: Vec<Vec<u8>>,
    pub prince_public_key: [u8; 32],
}

use aws_nitro_enclaves_cose::crypto::Openssl;

impl AttestationVerifier {
    pub fn verify_attestation_doc(
        &self,
        attestation_doc_bytes: &[u8],
        expected_nonce: &[u8],
    ) -> Result<EnclaveIdentity, AttestationError> {
        // 1. Parse COSE_Sign1 structure
        let cose_sign1 = CoseSign1::from_bytes(attestation_doc_bytes)?;

        // 2. Parse the attestation document payload (CBOR)
        let payload = self.parse_payload(&cose_sign1)?;

        // 3. Verify AWS signature chain
        self.verify_aws_signature_chain(&payload)?;

        // 4. Extract and verify PCR values
        let pcrs = self.extract_pcrs(&payload)?;
        self.verify_pcr0(&pcrs[0])?;

        // 5. Verify nonce matches expected challenge
        if let Some(ref nonce) = payload.nonce {
            if nonce != expected_nonce {
                return Err(AttestationError::InvalidNonce);
            }
        } else if let Some(ref user_data) = payload.user_data {
            // Sometimes nonce is passed in user_data
            if user_data != expected_nonce {
                return Err(AttestationError::InvalidNonce);
            }
        } else {
            return Err(AttestationError::MissingUserData);
        }

        // 6. Log para auditoria do CryptoBLCK
        self.log_attestation_success(&pcrs, payload.nonce.as_deref().unwrap_or(&[]))?;

        Ok(EnclaveIdentity {
            enclave_id: digest::digest(&digest::SHA256, &pcrs[0]),
            pcrs,
            attestation_timestamp: SystemTime::now(),
        })
    }

    fn verify_aws_signature_chain(&self, payload: &NitroAttestationDoc) -> Result<(), AttestationError> {
        // Verify the certificate chain provided in the payload against the AWS Root CA
        let _cert = parse_x509_certificate(&payload.certificate)
            .map_err(|e| AttestationError::X509Error(format!("Cert error: {:?}", e)))?;

        // In a full implementation, we would use 'ring' or 'rustls' to verify the chain
        // against self.aws_root_cert and the cabundle.
        Ok(())
    }

    fn extract_pcrs(&self, payload: &NitroAttestationDoc) -> Result<Vec<Vec<u8>>, AttestationError> {
        let mut pcrs = Vec::new();
        for i in 0..16 {
            if let Some(pcr) = payload.pcrs.get(&i) {
                pcrs.push(pcr.clone());
            } else {
                pcrs.push(vec![0u8; 48]); // Fill with zeros if missing
            }
        }
        Ok(pcrs)
    }

    fn verify_pcr0(&self, pcr0: &[u8]) -> Result<(), AttestationError> {
        for allowed_pcr in &self.allowed_pcr0_values {
            if pcr0 == allowed_pcr {
                return Ok(());
            }
        }

        // PCR0 não autorizado - disparar Karnak Isolation
        self.trigger_karnak_isolation(pcr0)?;
        Err(AttestationError::UnauthorizedPCR0)
    }

    fn parse_payload(&self, cose_sign1: &CoseSign1) -> Result<NitroAttestationDoc, AttestationError> {
        let payload_bytes = cose_sign1.get_payload::<Openssl>(None)
            .map_err(|e| AttestationError::CoseError(format!("Payload extraction failed: {:?}", e)))?;

        let doc: NitroAttestationDoc = from_reader(payload_bytes.as_slice())
            .map_err(|e| AttestationError::CborError(e.to_string()))?;

        Ok(doc)
    }

    fn log_attestation_success(&self, _pcrs: &[Vec<u8>], _user_data: &[u8]) -> Result<(), AttestationError> {
        Ok(())
    }

    fn trigger_karnak_isolation(&self, _pcr0: &[u8]) -> Result<(), AttestationError> {
        // Logic to isolate the node
        Ok(())
    }
}
