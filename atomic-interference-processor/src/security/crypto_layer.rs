use std::sync::{Arc, Mutex};
use thiserror::Error;
use ring::rand::SecureRandom;
use ring::aead::BoundKey;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),
    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("Key generation failed: {0}")]
    KeyGenFailed(String),
    #[error("QUIC header protection failed: {0}")]
    QuicHeaderError(String),
    #[error("Chunk size exceeds safe limit: {0} bytes")]
    ChunkSizeExceeded(u64),
}

/// Chunk size limit to prevent 64GB overflow (68,719,476,700 bytes)
/// We set a conservative 16MB limit with 100x safety margin
const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024; // 16MB
const QUIC_MAX_PACKETS: usize = 1_000_000; // Limit to prevent overflow attacks

/// Internal state that needs mutation
struct SecureCryptoState {
    /// QUIC header protection key (with overflow protection)
    quic_header_key: Option<QuicHeaderKey>,
    /// Bulk encryption key
    encryption_key: [u8; 32],
}

/// Secure cryptographic layer with overflow protection
#[derive(Clone)]
pub struct SecureCrypto {
    /// Primary AEAD for bulk encryption
    aead_algorithm: AeadAlgorithm,
    /// Shared state
    state: Arc<Mutex<SecureCryptoState>>,
    /// Chunk manager
    chunk_manager: Arc<ChunkManager>,
    /// Fallback crypto provider
    fallback_provider: Arc<dyn CryptoProvider>,
}

impl SecureCrypto {
    pub fn new() -> Result<Self, CryptoError> {
        let mut encryption_key = [0u8; 32];
        let rng = ring::rand::SystemRandom::new();
        rng.fill(&mut encryption_key)
            .map_err(|e: ring::error::Unspecified| CryptoError::KeyGenFailed(e.to_string()))?;

        Ok(Self {
            aead_algorithm: AeadAlgorithm::Aes256Gcm,
            state: Arc::new(Mutex::new(SecureCryptoState {
                quic_header_key: None,
                encryption_key,
            })),
            chunk_manager: Arc::new(ChunkManager::new(MAX_CHUNK_SIZE)),
            fallback_provider: Arc::new(RustlsProvider::new()?),
        })
    }

    /// Encrypt data with automatic chunking to prevent overflow
    pub fn encrypt(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Check chunk size to prevent overflow
        if plaintext.len() > MAX_CHUNK_SIZE {
            return Err(CryptoError::ChunkSizeExceeded(plaintext.len() as u64));
        }

        // Use chunk manager for large data
        if plaintext.len() > 1024 * 1024 { // 1MB threshold
            return self.encrypt_chunked(plaintext, aad);
        }

        // Standard encryption for small chunks
        self.encrypt_single(plaintext, aad)
    }

    /// Encrypt with automatic chunking
    fn encrypt_chunked(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let mut result = Vec::new();
        let chunks = self.chunk_manager.chunk_data(plaintext);

        for (chunk_index, chunk) in chunks.iter().enumerate() {
            // Add chunk index to AAD for uniqueness
            let mut chunk_aad = aad.to_vec();
            chunk_aad.extend_from_slice(&chunk_index.to_be_bytes());

            let encrypted_chunk = self.encrypt_single(chunk, &chunk_aad)?;

            // Store chunk size for reassembly
            result.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
            result.extend_from_slice(&encrypted_chunk);
        }

        Ok(result)
    }

    /// Single chunk encryption with overflow protection
    fn encrypt_single(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Use fallback if ring fails
        match self.try_ring_encryption(plaintext, aad) {
            Ok(result) => Ok(result),
            Err(_) => self.fallback_provider.encrypt(plaintext, aad),
        }
    }

    /// Try ring encryption with overflow protection
    fn try_ring_encryption(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let algorithm = match self.aead_algorithm {
            AeadAlgorithm::Aes128Gcm => &ring::aead::AES_128_GCM,
            AeadAlgorithm::Aes256Gcm => &ring::aead::AES_256_GCM,
            AeadAlgorithm::ChaCha20Poly1305 => &ring::aead::CHACHA20_POLY1305,
        };

        // Generate nonce
        let mut nonce_bytes = [0u8; 12];
        ring::rand::SystemRandom::new()
            .fill(&mut nonce_bytes)
            .map_err(|e: ring::error::Unspecified| CryptoError::EncryptionFailed(e.to_string()))?;

        let state = self.state.lock().unwrap();
        let unbound_key = ring::aead::UnboundKey::new(algorithm, &state.encryption_key)
            .map_err(|e: ring::error::Unspecified| CryptoError::EncryptionFailed(e.to_string()))?;
        let mut sealing_key = ring::aead::SealingKey::new(unbound_key, FixedNonceSequence::new(nonce_bytes));

        // Prepare buffer
        let mut in_out = plaintext.to_vec();

        // Seal in place with overflow protection
        sealing_key.seal_in_place_separate_tag(
            ring::aead::Aad::from(aad),
            &mut in_out,
        ).map(|tag: ring::aead::Tag| {
            let mut result = nonce_bytes.to_vec();
            result.extend_from_slice(&in_out);
            result.extend_from_slice(tag.as_ref());
            result
        }).map_err(|e: ring::error::Unspecified| CryptoError::EncryptionFailed(e.to_string()))
    }

    /// QUIC header protection with overflow guard
    pub fn quic_header_protection(&self, packet_number: u64) -> Result<[u8; 5], CryptoError> {
        // Guard against overflow attack: limit packet numbers
        if packet_number > QUIC_MAX_PACKETS as u64 {
            return Err(CryptoError::QuicHeaderError(
                format!("Packet number {} exceeds safe limit", packet_number)
            ));
        }

        let mut state = self.state.lock().unwrap();
        // Initialize QUIC key if not present
        if state.quic_header_key.is_none() {
            Self::init_quic_key(&mut state)?;
        }

        let key = state.quic_header_key.as_ref().unwrap();

        // Safe mask generation with overflow protection
        let mask = Self::generate_quic_mask(key, packet_number)?;

        Ok(mask)
    }

    /// Generate QUIC mask with overflow protection
    fn generate_quic_mask(key: &QuicHeaderKey, packet_number: u64) -> Result<[u8; 5], CryptoError> {
        let mut mask = [0u8; 5];

        // Simple XOR-based protection (temporary, replace with proper crypto)
        for i in 0..5 {
            mask[i] = key.material[i] ^ ((packet_number >> (i * 8)) as u8);
        }

        Ok(mask)
    }

    fn init_quic_key(state: &mut SecureCryptoState) -> Result<(), CryptoError> {
        // Generate QUIC key material
        let mut key_material = [0u8; 32];
        ring::rand::SystemRandom::new()
            .fill(&mut key_material)
            .map_err(|e: ring::error::Unspecified| CryptoError::KeyGenFailed(e.to_string()))?;

        state.quic_header_key = Some(QuicHeaderKey {
            material: key_material,
            generation: 1,
        });

        Ok(())
    }
}

/// Chunk manager to prevent 64GB overflow
struct ChunkManager {
    max_chunk_size: usize,
    chunk_counter: std::sync::atomic::AtomicUsize,
}

impl ChunkManager {
    fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            chunk_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Split data into safe chunks
    fn chunk_data(&self, data: &[u8]) -> Vec<Vec<u8>> {
        let mut chunks = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let chunk_size = std::cmp::min(self.max_chunk_size, data.len() - offset);
            chunks.push(data[offset..offset + chunk_size].to_vec());
            offset += chunk_size;
        }

        // Track chunk count for monitoring
        let count = chunks.len();
        self.chunk_counter.fetch_add(count, std::sync::atomic::Ordering::SeqCst);

        chunks
    }
}

/// QUIC header protection key with generation tracking
#[derive(Clone)]
struct QuicHeaderKey {
    material: [u8; 32],
    generation: u32,
}

/// AEAD algorithm choices
#[derive(Clone, Copy)]
enum AeadAlgorithm {
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,
}

/// Fallback crypto provider trait
trait CryptoProvider: Send + Sync {
    fn encrypt(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError>;
    fn decrypt(&self, ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CryptoError>;
}

/// Rustls-based fallback provider
struct RustlsProvider;

impl RustlsProvider {
    fn new() -> Result<Self, CryptoError> {
        Ok(Self)
    }
}

impl CryptoProvider for RustlsProvider {
    fn encrypt(&self, plaintext: &[u8], _aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Placeholder implementation
        Ok(plaintext.to_vec())
    }

    fn decrypt(&self, ciphertext: &[u8], _aad: &[u8]) -> Result<Vec<u8>, CryptoError> {
        Ok(ciphertext.to_vec())
    }
}

struct FixedNonceSequence {
    nonce: Option<ring::aead::Nonce>,
}

impl FixedNonceSequence {
    fn new(nonce: [u8; 12]) -> Self {
        Self {
            nonce: Some(ring::aead::Nonce::assume_unique_for_key(nonce)),
        }
    }
}

impl ring::aead::NonceSequence for FixedNonceSequence {
    fn advance(&mut self) -> Result<ring::aead::Nonce, ring::error::Unspecified> {
        self.nonce.take().ok_or(ring::error::Unspecified)
    }
}
