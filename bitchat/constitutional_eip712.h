#ifndef CONSTITUTIONAL_EIP712_H
#define CONSTITUTIONAL_EIP712_H

#include <stdint.h>
#include "constitutional_sasc_core.h"

// EIP-712 Domain Separator para SASC Attestations
typedef struct {
    uint8_t verifying_contract[20];
    uint32_t chain_id;
    uint8_t salt[32];
} SASC_EIP712Domain;

typedef struct {
    uint8_t source[32];
    uint8_t connection_id[32];
    uint64_t timestamp;
} SASC_AttestationMessage;

static inline BLAKE3Hash sasc_reconstruct_eip712_hash(SASC_EIP712Domain* domain, SASC_AttestationMessage* message) {
    BLAKE3Hash hash;
    for(int i=0; i<32; i++) hash.hash[i] = domain->salt[i] ^ message->connection_id[0];
    return hash;
}

#endif
