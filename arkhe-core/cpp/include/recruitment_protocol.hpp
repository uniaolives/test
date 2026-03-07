#pragma once

#include "identity.hpp"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace arkhe::network {

class RecruitmentProtocol {
public:
    struct VerificationResult {
        bool success;
        std::string reason;
    };

    static VerificationResult verify_recruitment_proof(const std::string& node_id,
                                                      const std::string& ed_pk,
                                                      const std::string& pq_pk,
                                                      const std::string& signature,
                                                      double phi_q) {
        // 1. Minimum coherence threshold (Miller Limit φ_q = 4.64)
        if (phi_q < 4.64) {
            return {false, "Insufficient coherence (phi_q < 4.64)"};
        }

        // 2. Verify hybrid signature
        // The message is the node_id + formatted phi_q
        std::string message = node_id + core::HybridIdentity::format_double(phi_q);
        auto hybrid_sig = core::HybridIdentity::Signature::deserialize(signature);

        if (!core::HybridIdentity::verify(message, hybrid_sig, ed_pk, pq_pk)) {
            return {false, "Invalid hybrid (post-quantum) cryptographic signature"};
        }

        return {true, "Identity Hardened and Verified (PQC)"};
    }
};

} // namespace arkhe::network
