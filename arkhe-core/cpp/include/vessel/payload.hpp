#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace Arkhe::Vessel {

/**
 * The SatoshiPayload represents the hidden cargo of the Satoshi Vessel.
 * It contains the intention (message) and the cryptographic proof of target.
 */
struct SatoshiPayload {
    std::string message;
    std::string target_address;     // Coordinates: satoshi@anonymousspeech.com
    uint64_t target_timestamp;      // Anchor: 1231006505 (Genesis Block)
    std::vector<uint8_t> dilithium_signature; // Proof of Identity
};

} // namespace Arkhe::Vessel
