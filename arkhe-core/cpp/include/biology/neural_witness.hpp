#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cstdint>

namespace Arkhe::Biology {

    struct NeuralWitness {
        double hrv;          // Heart Rate Variability
        double vagal_tone;   // Vagal tone (parasympathetic nervous system)
        double phi_q;        // Quantum coherence metric

        std::string get_identity() const {
            return "neuromancer://biological_oracle_0x1";
        }

        std::vector<uint8_t> encode_to_payload() const {
            // Simple encoding of neural state for ZK witness
            std::string data = std::to_string(hrv) + ":" + std::to_string(vagal_tone);
            return std::vector<uint8_t>(data.begin(), data.end());
        }
    };

} // namespace Arkhe::Biology
