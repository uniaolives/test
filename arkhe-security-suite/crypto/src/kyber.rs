use pqcrypto_kyber::kyber1024::*;
pub use pqcrypto_kyber::kyber1024::{PublicKey, SecretKey, Ciphertext, SharedSecret};

pub fn generate_keypair() -> (PublicKey, SecretKey) {
    keypair()
}

pub fn encapsulate_key(pk: &PublicKey) -> (Ciphertext, SharedSecret) {
    let (ss, ct) = encapsulate(pk);
    (ct, ss)
}

pub fn decapsulate_key(ct: &Ciphertext, sk: &SecretKey) -> SharedSecret {
    decapsulate(ct, sk)
}
