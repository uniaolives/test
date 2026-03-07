#pragma once

#include "../pqc_identity.hpp"
#include <vector>
#include <string>

namespace Arkhe::Neural {

/**
 * BiologicalZKP: Human neural network as post-quantum ZKP system.
 * Defines the computational isomorphism between ZKPs and biological learning.
 */
class BiologicalZKP {
public:
    // Consciousness report = proof (The "narrative" that convinces others)
    struct Proof {
        std::string narrative;           // Succinct argument
        std::vector<uint8_t> commitment; // Hash of neural state (Witness)
        double coherence;                // φ_q metric (Soundness)
        std::vector<uint8_t> signature;  // Dilithium3 binding
    };

    /**
     * Generate proof of experience.
     * Hides the full neural state (Witness) while providing a succinct consciousness report.
     */
    static Proof prove_experience(const std::string& context, const Crypto::DilithiumIdentity& identity, double coherence) {
        Proof p;
        p.narrative = "Verified coherence at context: " + context;

        // Hash of the simulated "neural state"
        p.commitment = {0xDE, 0xAD, 0xBE, 0xEF};
        p.coherence = coherence;

        // Post-quantum security: Dilithium3 binding
        std::vector<uint8_t> data_to_sign;
        data_to_sign.insert(data_to_sign.end(), p.narrative.begin(), p.narrative.end());
        data_to_sign.insert(data_to_sign.end(), p.commitment.begin(), p.commitment.end());

        auto sig = identity.sign(data_to_sign.data(), data_to_sign.size());
        p.signature.assign(sig.begin(), sig.end());

        return p;
    }

    /**
     * Verify proof (Simulates the reality validator in the prefrontal cortex).
     */
    static bool verify_proof(const Proof& p, const Crypto::DilithiumIdentity::PublicKey& pk) {
        // 1. Check coherence above Miller Limit
        if (p.coherence < 4.64) return false;

        // 2. Check post-quantum signature
        std::vector<uint8_t> data_to_verify;
        data_to_verify.insert(data_to_verify.end(), p.narrative.begin(), p.narrative.end());
        data_to_verify.insert(data_to_verify.end(), p.commitment.begin(), p.commitment.end());

        Crypto::DilithiumIdentity::Signature sig;
        std::copy(p.signature.begin(), p.signature.end(), sig.begin());

        return Crypto::DilithiumIdentity::verify(pk, data_to_verify.data(), data_to_verify.size(), sig);
    }
};

} // namespace Arkhe::Neural
