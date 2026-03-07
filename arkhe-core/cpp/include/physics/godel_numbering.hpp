// arkhe-core/cpp/include/physics/godel_numbering.hpp
// Encoding unstable periodic orbits using bitsets of prime factors to avoid overflow

#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <cmath>
#include <bitset>
#include <algorithm>

namespace Arkhe::Godel {

// Maximum number of primes for encoding features
static constexpr size_t MAX_GODEL_DIM = 1024;

// Gödel signature for a ghost orbit = bitset where each bit represents a prime factor p_i
class GhostGodelSignature {
    static inline std::vector<uint64_t> prime_basis_;  // Shared first 1024 primes
    std::bitset<MAX_GODEL_DIM> signature_; // Active prime factors

public:
    GhostGodelSignature(const std::vector<double>& embedding) {
        if (prime_basis_.empty()) {
            generate_primes(MAX_GODEL_DIM);
        }
        for (size_t i = 0; i < embedding.size() && i < MAX_GODEL_DIM; ++i) {
            if (std::abs(embedding[i]) > threshold_) {
                signature_.set(i);
            }
        }
    }

    // Returns the signature bitset
    const std::bitset<MAX_GODEL_DIM>& get_signature() const {
        return signature_;
    }

    // Two orbits are "the same ghost" if signatures share many factors (Jaccard similarity)
    static double similarity(const GhostGodelSignature& a,
                            const GhostGodelSignature& b) {
        auto intersection = (a.signature_ & b.signature_).count();
        auto union_set = (a.signature_ | b.signature_).count();
        if (union_set == 0) return 1.0;
        return static_cast<double>(intersection) / union_set;
    }

private:
    static constexpr double threshold_ = 0.1;

    static void generate_primes(size_t n) {
        uint64_t num = 2;
        while (prime_basis_.size() < n) {
            bool is_prime = true;
            for (uint64_t i = 2; i <= std::sqrt(num); ++i) {
                if (num % i == 0) {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime) {
                prime_basis_.push_back(num);
            }
            num++;
        }
    }
};

// The "ghost cluster" is a set of orbits with shared Gödel factors
class GhostCluster {
    std::vector<GhostGodelSignature> members_;

public:
    void add(const GhostGodelSignature& ghost) {
        // Check if ghost belongs to this cluster (high similarity)
        for (const auto& member : members_) {
            if (GhostGodelSignature::similarity(ghost, member) > 0.5) {
                members_.push_back(ghost);
                return;
            }
        }
        // Or start new cluster
        if (members_.empty()) members_.push_back(ghost);
    }

    // The cluster "exists" as a stable ghost if members share common factors
    bool is_stable() const {
        if (members_.size() < 10) return false;

        // Compute cluster core signature = intersection of all members
        std::bitset<MAX_GODEL_DIM> cluster_core;
        cluster_core.set(); // All bits to 1
        for (const auto& m : members_) {
            cluster_core &= m.get_signature();
        }

        // Stability = many shared prime factors (active bits in intersection)
        return cluster_core.count() > 5;  // Threshold for "real" ghost
    }
};

} // namespace Arkhe::Godel
