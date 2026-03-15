use ed25519_dalek::{Signer, Signature, SigningKey};
use rand::rngs::OsRng;

pub struct SignatureEngine {
    key: SigningKey,
use ed25519_dalek::{Keypair, Signer, Signature};
use rand::rngs::OsRng;

pub struct SignatureEngine {
    keypair: Keypair,
}

impl SignatureEngine {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let key = SigningKey::generate(&mut csprng);
        Self { key }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.key.sign(message)
        let keypair = Keypair::generate(&mut csprng);
        Self { keypair }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.keypair.sign(message)
    }
}
