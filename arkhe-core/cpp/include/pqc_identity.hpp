#pragma once

#include <oqs/oqs.h>
#include <array>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

namespace Arkhe::Crypto {

    // ============================================================
    // DILITHIUM3: ASSINATURAS DIGITAIS PÓS-QUÂNTICAS
    // ============================================================

    class DilithiumIdentity {
    public:
        // Dilithium3 parameters (fixed sizes for NIST Level 3)
        static constexpr size_t PUBLIC_KEY_SIZE = 1952;
        static constexpr size_t SECRET_KEY_SIZE = 4016;
        static constexpr size_t SIGNATURE_SIZE  = 3293;

        using PublicKey = std::array<uint8_t, PUBLIC_KEY_SIZE>;
        using SecretKey = std::array<uint8_t, SECRET_KEY_SIZE>;
        using Signature = std::array<uint8_t, SIGNATURE_SIZE>;

    private:
        PublicKey pk_;
        SecretKey sk_;

        OQS_SIG *sig_;

    public:
        DilithiumIdentity() {
            sig_ = OQS_SIG_new(OQS_SIG_alg_dilithium_3);
            if (!sig_) throw std::runtime_error("Failed to initialize Dilithium3 via liboqs");

            OQS_STATUS rc = OQS_SIG_keypair(sig_, pk_.data(), sk_.data());
            if (rc != OQS_SUCCESS) {
                OQS_SIG_free(sig_);
                throw std::runtime_error("Dilithium3 keypair generation failed");
            }
        }

        ~DilithiumIdentity() {
            if (sig_) {
                OQS_MEM_cleanse(sk_.data(), SECRET_KEY_SIZE);
                OQS_SIG_free(sig_);
            }
        }

        // Move semantics (prevent secret key copy)
        DilithiumIdentity(DilithiumIdentity&& other) noexcept : pk_(std::move(other.pk_)), sk_(std::move(other.sk_)), sig_(other.sig_) {
            other.sig_ = nullptr;
        }

        DilithiumIdentity& operator=(DilithiumIdentity&& other) noexcept {
            if (this != &other) {
                if (sig_) {
                    OQS_MEM_cleanse(sk_.data(), SECRET_KEY_SIZE);
                    OQS_SIG_free(sig_);
                }
                pk_ = std::move(other.pk_);
                sk_ = std::move(other.sk_);
                sig_ = other.sig_;
                other.sig_ = nullptr;
            }
            return *this;
        }

        // Sign message
        Signature sign(const uint8_t* message, size_t message_len) const {
            Signature sig_data;
            size_t sig_len;
            OQS_STATUS rc = OQS_SIG_sign(sig_, sig_data.data(), &sig_len, message, message_len, sk_.data());
            if (rc != OQS_SUCCESS || sig_len != SIGNATURE_SIZE) {
                throw std::runtime_error("Post-quantum signature creation failed");
            }
            return sig_data;
        }

        // Verify signature
        static bool verify(const PublicKey& pk, const uint8_t* message, size_t message_len, const Signature& sig_data) {
            OQS_SIG *verifier = OQS_SIG_new(OQS_SIG_alg_dilithium_3);
            if (!verifier) return false;
            OQS_STATUS rc = OQS_SIG_verify(verifier, message, message_len, sig_data.data(), SIGNATURE_SIZE, pk.data());
            OQS_SIG_free(verifier);
            return (rc == OQS_SUCCESS);
        }

        const PublicKey& public_key() const { return pk_; }
    };
}
