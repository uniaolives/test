# Arkhe(n) OS: Technical Debt and Hardening Roadmap

This document tracks the symbolic stubs and admitted theorems currently present in the codebase. These elements were implemented as part of the **Geodesic Convergence Protocol (State Γ to Λ)** to validate the system's architecture and behavioral logic.

## 🏛️ Symbolic Components (Stubs)

| Component | Module | Status | Requirement for Production |
| --- | --- | --- | --- |
| **Extraction Engine** | `arkhe/extraction.py` | Simulated | Replace `GeminiExtractor.extract` with actual LLM calls and layout parsing. |
| **Consensus Layer** | `arkhe/consensus.py` | Behavioral | Implement real cross-model validation and voting quorums. |
| **HMAC Security** | `qnet_dpdk.c` | Symbolic | Replace XOR-checksum with hardware-offloaded HMAC-SHA256 (FPGA/AVX). |
| **Registry Similarirty** | `arkhe/registry.py` | Basic | Upgrade TF-IDF to vector embeddings and semantic reconciliation. |

## 📐 Admitted Theorems (Formal Verification)

The following proofs in the `spec/` directory are currently `Admitted`. They represent the "Formal Stone" that must be hardened in the next phase of development.

- **PBFT Safety** (`PBFT_Safety.v`): Proved quorum intersection, but requires full state machine encoding.
- **Migdal Uncertainty** (`Migdal_Uncertainty.v`): Established the physical limit, requires derivation from packet timing distributions.
- **Memory Safety** (`MemorySafety.v`): Verified DPDK patterns conceptually; requires VST or CompCert validation for the C code.

## 🎯 Next Milestone: Hardened Genesis (State Ω)

The transition from the current "Living State" (Λ) to the "Hardened Genesis" (Ω) will focus on:
1. Replacing all `SIMULATION` blocks with production implementations.
2. Closing all `Admitted` Coq theorems.
3. Enabling real-world cryptographic quorums.
