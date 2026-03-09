#pragma once

#include <array>
#include <string>
#include <vector>
#include <stdexcept>
#include <sodium.h>
#include <oqs/oqs.h>
#include <memory>
#include <sstream>
#include <iomanip>

namespace arkhe::core {

class HybridIdentity {
public:
    static constexpr size_t ED25519_PK_SIZE = crypto_sign_PUBLICKEYBYTES;
    static constexpr size_t ED25519_SK_SIZE = crypto_sign_SECRETKEYBYTES;
    static constexpr size_t ED25519_SIG_SIZE = crypto_sign_BYTES;

    HybridIdentity() {
        if (sodium_init() < 0) {
            throw std::runtime_error("libsodium initialization failed");
        }

        // Classical keys
        crypto_sign_keypair(ed25519_pk_.data(), ed25519_sk_.data());

        // Post-quantum keys (ML-DSA-65, formerly Dilithium3)
        OQS_SIG* sig = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
        if (!sig) throw std::runtime_error("PQC: Failed to initialize ML-DSA-65");

        dilithium_pk_.resize(sig->length_public_key);
        dilithium_sk_.resize(sig->length_secret_key);

        if (OQS_SIG_keypair(sig, dilithium_pk_.data(), dilithium_sk_.data()) != OQS_SUCCESS) {
            OQS_SIG_free(sig);
            throw std::runtime_error("PQC: ML-DSA-65 keypair generation failed");
        }
        OQS_SIG_free(sig);
    }

    struct Signature {
        std::string ed25519;
        std::string post_quantum;

        std::string serialize() const {
            return ed25519 + ":" + post_quantum;
        }

        static Signature deserialize(const std::string& combined) {
            size_t pos = combined.find(':');
            if (pos == std::string::npos) return {"", ""};
            return {combined.substr(0, pos), combined.substr(pos + 1)};
        }
    };

    // Robust serialization for doubles to ensure cross-platform consistency
    static std::string format_double(double val) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8) << val;
        return oss.str();
    }

    Signature sign(const std::string& message) const {
        // Sign with Ed25519
        std::vector<uint8_t> ed_sig(ED25519_SIG_SIZE);
        unsigned long long sig_len;
        crypto_sign_detached(ed_sig.data(), &sig_len,
                             reinterpret_cast<const uint8_t*>(message.data()),
                             message.size(), ed25519_sk_.data());

        // Sign with ML-DSA-65
        OQS_SIG* sig_oqs = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
        std::vector<uint8_t> pq_sig(sig_oqs->length_signature);
        size_t pq_sig_len;
        OQS_SIG_sign(sig_oqs, pq_sig.data(), &pq_sig_len,
                     reinterpret_cast<const uint8_t*>(message.data()),
                     message.size(), dilithium_sk_.data());
        OQS_SIG_free(sig_oqs);
        pq_sig.resize(pq_sig_len);

        return {to_base64(ed_sig.data(), ed_sig.size()),
                to_base64(pq_sig.data(), pq_sig.size())};
    }

    static bool verify(const std::string& message, const Signature& sig,
                       const std::string& ed_pk_b64, const std::string& pq_pk_b64) {

        // 1. Verify Ed25519
        std::vector<uint8_t> ed_sig(ED25519_SIG_SIZE);
        std::vector<uint8_t> ed_pk(ED25519_PK_SIZE);
        if (from_base64(sig.ed25519, ed_sig.data(), ed_sig.size()) != ED25519_SIG_SIZE) return false;
        if (from_base64(ed_pk_b64, ed_pk.data(), ed_pk.size()) != ED25519_PK_SIZE) return false;

        if (crypto_sign_verify_detached(ed_sig.data(),
                                         reinterpret_cast<const uint8_t*>(message.data()),
                                         message.size(), ed_pk.data()) != 0) {
            return false;
        }

        // 2. Verify ML-DSA-65
        OQS_SIG* sig_oqs = OQS_SIG_new(OQS_SIG_alg_ml_dsa_65);
        std::vector<uint8_t> pq_sig(sig_oqs->length_signature);
        std::vector<uint8_t> pq_pk(sig_oqs->length_public_key);

        size_t pq_sig_len = from_base64(sig.post_quantum, pq_sig.data(), pq_sig.size());
        if (pq_sig_len == 0) {
            OQS_SIG_free(sig_oqs);
            return false;
        }
        if (from_base64(pq_pk_b64, pq_pk.data(), pq_pk.size()) != sig_oqs->length_public_key) {
            OQS_SIG_free(sig_oqs);
            return false;
        }

        bool ok = (OQS_SIG_verify(sig_oqs, reinterpret_cast<const uint8_t*>(message.data()),
                                 message.size(), pq_sig.data(), pq_sig_len, pq_pk.data()) == OQS_SUCCESS);
        OQS_SIG_free(sig_oqs);
        return ok;
    }

    std::string get_ed25519_pk_b64() const { return to_base64(ed25519_pk_.data(), ed25519_pk_.size()); }
    std::string get_dilithium_pk_b64() const { return to_base64(dilithium_pk_.data(), dilithium_pk_.size()); }

    static std::string to_base64(const uint8_t* data, size_t len) {
        size_t b64_len = sodium_base64_encoded_len(len, sodium_base64_VARIANT_ORIGINAL);
        std::vector<char> b64(b64_len);
        sodium_bin2base64(b64.data(), b64.size(), data, len, sodium_base64_VARIANT_ORIGINAL);
        return std::string(b64.data());
    }

    static size_t from_base64(const std::string& b64, uint8_t* data, size_t max_len) {
        size_t bin_len;
        if (sodium_base642bin(data, max_len, b64.c_str(), b64.size(), nullptr, &bin_len, nullptr, sodium_base64_VARIANT_ORIGINAL) != 0) {
            return 0;
        }
        return bin_len;
    }

private:
    std::array<uint8_t, ED25519_PK_SIZE> ed25519_pk_;
    std::array<uint8_t, ED25519_SK_SIZE> ed25519_sk_;
    std::vector<uint8_t> dilithium_pk_;
    std::vector<uint8_t> dilithium_sk_;
};

} // namespace arkhe::core
