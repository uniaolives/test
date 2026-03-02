// arkhe-axos-instaweb/src/axos/crypto.rs
use pqcrypto_kyber::kyber768;
use pqcrypto_sphincsplus::sphincssha256128fsimple;

pub struct PqcKeyPair {
    pub kem_pk: kyber768::PublicKey,
    pub kem_sk: kyber768::SecretKey,
    pub sign_pk: sphincssha256128fsimple::PublicKey,
    pub sign_sk: sphincssha256128fsimple::SecretKey,
}

impl PqcKeyPair {
    pub fn generate() -> Self {
        let (kem_pk, kem_sk) = kyber768::keypair();
        let (sign_pk, sign_sk) = sphincssha256128fsimple::keypair();
        Self {
            kem_pk,
            kem_sk,
            sign_pk,
            sign_sk,
        }
    }

    pub fn sign(&self, message: &[u8]) -> sphincssha256128fsimple::DetachedSignature {
        sphincssha256128fsimple::detached_sign(message, &self.sign_sk)
    }

    pub fn verify(&self, message: &[u8], signature: &sphincssha256128fsimple::DetachedSignature) -> bool {
        sphincssha256128fsimple::verify_detached_signature(signature, message, &self.sign_pk).is_ok()
    }

    pub fn encrypt(&self) -> (kyber768::SharedSecret, kyber768::Ciphertext) {
        kyber768::encapsulate(&self.kem_pk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pqc_sign_verify() {
        let keypair = PqcKeyPair::generate();
        let msg = b"Arkhe Protocol Constitution";
        let sig = keypair.sign(msg);
        assert!(keypair.verify(msg, &sig));
        assert!(!keypair.verify(b"Corrupted message", &sig));
    }
}
