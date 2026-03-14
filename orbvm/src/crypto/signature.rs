use ed25519_dalek::{Keypair, Signer, Signature};
use rand::rngs::OsRng;

pub struct SignatureEngine {
    keypair: Keypair,
}

impl SignatureEngine {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        Self { keypair }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.keypair.sign(message)
    }
}
