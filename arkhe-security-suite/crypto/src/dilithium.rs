use pqcrypto_dilithium::dilithium5::*;
use pqcrypto_traits::sign::{PublicKey as _, SecretKey as _, DetachedSignature as _};

pub fn generate_keypair() -> (PublicKey, SecretKey) {
    keypair()
}

pub fn sign(msg: &[u8], sk: &SecretKey) -> DetachedSignature {
    detached_sign(msg, sk)
}

pub fn verify(msg: &[u8], sig: &DetachedSignature, pk: &PublicKey) -> Result<(), pqcrypto_traits::sign::VerificationError> {
    verify_detached_signature(sig, msg, pk)
}
