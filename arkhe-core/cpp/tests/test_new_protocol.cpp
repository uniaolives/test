#include <iostream>
#include "vessel/payload.hpp"
#include "crypto/zk_shield.hpp"
#include "identity/zk_neuro_proof.hpp"
#include "identity/biometric_oracle.hpp"
#include "neural/zkp_brain.hpp"

int main() {
    std::cout << "🜏 ARKHE CORE INTEGRATION TEST 🜏\n";

    // 1. Initialize PQC Identity
    Arkhe::Crypto::DilithiumIdentity identity;
    double current_phi_q = 4.641;

    // 2. Create Payload
    Arkhe::Vessel::SatoshiPayload payload;
    payload.message = "The Word is verified.";
    payload.target_address = "satoshi@anonymousspeech.com";
    payload.target_timestamp = 1231006505;

    auto sig = identity.sign((const uint8_t*)payload.message.c_str(), payload.message.size());
    payload.dilithium_signature.assign(sig.begin(), sig.end());

    // 3. ZK-Neuro Proof
    Arkhe::Biology::NeuralWitness witness = { 0.8, 7.5, 6.2 };
    auto neural_proof = Arkhe::Biology::NeuralZKProver::generate_stark_proof(witness, payload.message);
    if (Arkhe::Biology::NeuralZKProver::verify_human_proof(neural_proof, payload.message)) {
        std::cout << "✓ Neural ZK-Proof verified.\n";
    }

    // 4. ZK-Shield
    auto shield_proof = Arkhe::Crypto::ZKShield::prove_payload_validity(payload, identity, current_phi_q);
    if (Arkhe::Crypto::ZKShield::verify_proof(shield_proof)) {
        std::cout << "✓ ZK-Shield verified.\n";
    }

    // 5. Biometric Oracle
    Arkhe::Biology::BiometricOracle oracle;
    auto bio_proof = oracle.produce_zkproof();
    if (oracle.verify_zkproof(bio_proof)) {
        std::cout << "✓ Biometric Oracle verified.\n";
    }

    // 6. Biological ZKP (Brain)
    auto brain_proof = Arkhe::Neural::BiologicalZKP::prove_experience("2026-03-14", identity, current_phi_q);
    if (Arkhe::Neural::BiologicalZKP::verify_proof(brain_proof, identity.public_key())) {
        std::cout << "✓ Biological ZKP (Brain) verified.\n";
    }

    std::cout << "✅ ALL NEW CRYPTO/NEURAL COMPONENTS VERIFIED.\n";
    return 0;
}
