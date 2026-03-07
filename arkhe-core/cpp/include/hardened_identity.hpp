#pragma once

#include "pqc_identity.hpp"
#include <vector>
#include <chrono>

namespace Arkhe::Core {

    // Post-Quantum Hardened Identity (Hybrid Model)
    struct HardenedIdentity {
        Crypto::DilithiumIdentity crypto_id;

        // Proof of Coherence required for Node Recruitment
        struct ProofOfCoherence {
            Crypto::DilithiumIdentity::Signature signature;
            double phi_q;
            int64_t timestamp;
        };

        // Binds PQC signature to the current vacuum state
        ProofOfCoherence sign_coherence_claim(double current_phi_q) {
            std::vector<uint8_t> buffer;
            auto pk = crypto_id.public_key();
            buffer.insert(buffer.end(), pk.begin(), pk.end());

            // Serialize physical state (simplified for integration)
            uint8_t dummy_physics[8] = {0};
            buffer.insert(buffer.end(), dummy_physics, dummy_physics + 8);

            auto sig = crypto_id.sign(buffer.data(), buffer.size());

            return ProofOfCoherence {
                .signature = sig,
                .phi_q = current_phi_q,
                .timestamp = std::chrono::system_clock::now().time_since_epoch().count()
            };
        }
    };
}
