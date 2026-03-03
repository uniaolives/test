use pqcrypto_kyber::kyber1024::{
    self,
    PublicKey as KyberPublicKey,
    SecretKey as KyberSecretKey,
    Ciphertext as KyberCiphertext,
    SharedSecret as KyberSharedSecret
};
use pqcrypto_dilithium::dilithium5::{
    self,
    PublicKey as DilithiumPublicKey,
    SecretKey as DilithiumSecretKey,
    DetachedSignature as DilithiumSignature
};
use pqcrypto_traits::kem::{Ciphertext, SharedSecret};
use pqcrypto_traits::sign::{PublicKey, SecretKey, DetachedSignature};

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use anyhow::{Result, anyhow};

/// Par de chaves de longo prazo (Kyber + Dilithium) para um nó.
#[derive(Clone)]
pub struct NodeKeys {
    /// Chave secreta Dilithium (para assinar handovers).
    pub dilithium_secret: DilithiumSecretKey,
    /// Chave pública Dilithium.
    pub dilithium_public: DilithiumPublicKey,
    /// Chave secreta Kyber (para estabelecer chaves de sessão).
    pub kyber_secret: KyberSecretKey,
    /// Chave pública Kyber.
    pub kyber_public: KyberPublicKey,
}

impl NodeKeys {
    /// Gera um novo par de chaves.
    pub fn generate() -> Self {
        let (dilithium_public, dilithium_secret) = dilithium5::keypair();
        let (kyber_public, kyber_secret) = kyber1024::keypair();
        NodeKeys {
            dilithium_secret,
            dilithium_public,
            kyber_secret,
            kyber_public,
        }
    }
}

/// Chave de sessão estabelecida via Kyber KEM.
pub struct SessionKey {
    pub key: [u8; 32], // AES-256 key
    pub ciphertext: Vec<u8>, // para envio ao peer
}

impl SessionKey {
    /// Estabelece uma chave de sessão a partir da chave pública do peer.
    pub fn encapsulate(peer_public: &KyberPublicKey) -> Result<Self> {
        let (ciphertext, shared_secret) = kyber1024::encapsulate(peer_public);
        // shared_secret é um vetor de bytes; usamos os primeiros 32 para AES
        let mut key = [0u8; 32];
        key.copy_from_slice(&shared_secret.as_bytes()[..32]);
        Ok(SessionKey { key, ciphertext: ciphertext.as_bytes().to_vec() })
    }

    /// Decapsula a chave de sessão a partir do ciphertext.
    pub fn decapsulate(ciphertext: &[u8], secret: &KyberSecretKey) -> Result<Self> {
        let ciphertext_obj = KyberCiphertext::from_bytes(ciphertext)
            .map_err(|e| anyhow!("Falha ao converter ciphertext: {:?}", e))?;
        let shared_secret = kyber1024::decapsulate(&ciphertext_obj, secret);
        let mut key = [0u8; 32];
        key.copy_from_slice(&shared_secret.as_bytes()[..32]);
        Ok(SessionKey { key, ciphertext: ciphertext.to_vec() })
    }
}

/// Assina uma mensagem com a chave secreta Dilithium.
pub fn sign_message(msg: &[u8], secret: &DilithiumSecretKey) -> Vec<u8> {
    dilithium5::detached_sign(msg, secret).as_bytes().to_vec()
}

/// Verifica uma assinatura com a chave pública.
pub fn verify_signature(msg: &[u8], signature: &[u8], public: &DilithiumPublicKey) -> bool {
    let sig = match DilithiumSignature::from_bytes(signature) {
        Ok(s) => s,
        Err(_) => return false,
    };
    dilithium5::verify_detached_signature(&sig, msg, public).is_ok()
}

/// Criptografa um payload com AES-256-GCM usando a chave de sessão.
pub fn encrypt_payload(payload: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(key.into());
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher.encrypt(nonce, payload)
        .map_err(|e| anyhow!("Falha na criptografia: {:?}", e))?;
    // Retorna nonce + ciphertext
    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decriptografa um payload (nonce + ciphertext) com a chave de sessão.
pub fn decrypt_payload(encrypted: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    if encrypted.len() < 12 {
        return Err(anyhow!("Dados muito curtos"));
    }
    let (nonce_bytes, ciphertext) = encrypted.split_at(12);
    let cipher = Aes256Gcm::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher.decrypt(nonce, ciphertext)
        .map_err(|e| anyhow!("Falha na decriptografia: {:?}", e))?;
    Ok(plaintext)
}
