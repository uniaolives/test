use pqcrypto_kyber::kyber512;
use pqcrypto_dilithium::dilithium3;
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::kem::{PublicKey as KemPk, SecretKey as KemSk, Ciphertext as KemCt, SharedSecret as KemSs};
use pqcrypto_traits::sign::{PublicKey as SigPk, SecretKey as SigSk, SignedMessage as SigSm, DetachedSignature as SigDs};
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Key error: {0}")]
    KeyError(String),
    #[error("Ciphertext error: {0}")]
    CiphertextError(String),
    #[error("Signature error: {0}")]
    SignatureError(String),
    #[error("Algorithm not available")]
    AlgorithmNotAvailable,
    #[error("Invalid ciphertext")]
    InvalidCiphertext,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    #[error("Random error: {0}")]
    RandomError(String),
    #[error("Lock error")]
    LockError,
    #[error("Certificate expired")]
    CertificateExpired,
    #[error("Invalid issuer")]
    InvalidIssuer,
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Internal error: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NistSecurityLevel {
    Level1,
    Level2,
    Level3,
    Level5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PqcAlgorithm {
    Kyber512,
    Kyber768,
    Kyber1024,
    Dilithium2,
    Dilithium3,
    Dilithium5,
    Falcon512,
    Falcon1024,
    SphincsPlus128FSimple,
    Ed25519,
    P384,
}

pub struct KemKeyPair {
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub algorithm: PqcAlgorithm,
}

pub struct SignatureKeyPair {
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub algorithm: PqcAlgorithm,
}

pub trait KemScheme: Send + Sync {
    fn generate_keypair(&self) -> Result<KemKeyPair, CryptoError>;
    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;
    fn decapsulate(&self, ciphertext: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn public_key_size(&self) -> usize;
    fn security_level(&self) -> NistSecurityLevel;
}

pub struct Kyber512Kem;
impl KemScheme for Kyber512Kem {
    fn generate_keypair(&self) -> Result<KemKeyPair, CryptoError> {
        let (pk, sk) = kyber512::keypair();
        Ok(KemKeyPair {
            public_key: pk.as_bytes().to_vec(),
            private_key: sk.as_bytes().to_vec(),
            algorithm: PqcAlgorithm::Kyber512,
        })
    }
    fn encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        let pk = kyber512::PublicKey::from_bytes(public_key)
            .map_err(|e| CryptoError::KeyError(format!("{:?}", e)))?;
        let (ct, ss) = kyber512::encapsulate(&pk);
        Ok((ct.as_bytes().to_vec(), ss.as_bytes().to_vec()))
    }
    fn decapsulate(&self, ciphertext: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let sk = kyber512::SecretKey::from_bytes(private_key)
            .map_err(|e| CryptoError::KeyError(format!("{:?}", e)))?;
        let ct = kyber512::Ciphertext::from_bytes(ciphertext)
            .map_err(|e| CryptoError::CiphertextError(format!("{:?}", e)))?;
        let ss = kyber512::decapsulate(&ct, &sk);
        Ok(ss.as_bytes().to_vec())
    }
    fn public_key_size(&self) -> usize { kyber512::public_key_bytes() }
    fn security_level(&self) -> NistSecurityLevel { NistSecurityLevel::Level1 }
}

pub trait SignatureScheme: Send + Sync {
    fn generate_keypair(&self) -> Result<SignatureKeyPair, CryptoError>;
    fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn verify(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, CryptoError>;
    fn signature_size(&self) -> usize;
    fn security_level(&self) -> NistSecurityLevel;
}

pub struct Dilithium3Signature;
impl SignatureScheme for Dilithium3Signature {
    fn generate_keypair(&self) -> Result<SignatureKeyPair, CryptoError> {
        let (pk, sk) = dilithium3::keypair();
        Ok(SignatureKeyPair {
            public_key: pk.as_bytes().to_vec(),
            private_key: sk.as_bytes().to_vec(),
            algorithm: PqcAlgorithm::Dilithium3,
        })
    }
    fn sign(&self, message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let sk = dilithium3::SecretKey::from_bytes(private_key)
            .map_err(|e| CryptoError::KeyError(format!("{:?}", e)))?;
        let sm = dilithium3::sign(message, &sk);
        Ok(sm.as_bytes().to_vec())
    }
    fn verify(&self, _message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, CryptoError> {
        let pk = dilithium3::PublicKey::from_bytes(public_key)
            .map_err(|e| CryptoError::KeyError(format!("{:?}", e)))?;
        let sig = dilithium3::SignedMessage::from_bytes(signature)
            .map_err(|e| CryptoError::SignatureError(format!("{:?}", e)))?;
        Ok(dilithium3::open(&sig, &pk).is_ok())
    }
    fn signature_size(&self) -> usize { dilithium3::signature_bytes() }
    fn security_level(&self) -> NistSecurityLevel { NistSecurityLevel::Level2 }
}
