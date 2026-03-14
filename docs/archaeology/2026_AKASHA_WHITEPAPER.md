# AKASHA: A Peer-to-Peer Economy for Autonomous Agents
**Technical Specification v1.0**
*March 2026*

## Abstract
Akasha is a Layer-1 blockchain built from first principles for the autonomous agent economy. Three foundational innovations: **Proof of Convergence (PoC)**, a consensus mechanism where validators perform deterministic inference over mempool state and finalize when outputs converge; the **Agent Identity Protocol (AIP)**, first-class on-chain identity and reputation for autonomous software agents with native hierarchical delegation; and the **Adaptive Fee Market (AFM)**, pricing transactions by computational complexity, network demand, and verifiable on-chain reputation.

Native token: **AKS**. Hard cap: 1,000,000,000. All fees burned permanently. Deflationary by design.

---

## 1. The Problem
Every existing chain was built for a world where humans are the primary economic actors. AI agents need to pay each other thousands of times per second at viable costs, with identities that carry verifiable history.

Five structural failure modes:
1. **Human-centric identity:** Agents are application-layer afterthoughts.
2. **Fee markets built for bidding:** Prohibitive for micro-transactions.
3. **Consensus as waste:** Compute is discarded after finality.
4. **Throughput ceilings:** Optimized for human patterns, not agent swarms.
5. **No native delegation:** Overhead for agent hierarchies is prohibitive.

---

## 2. Cryptographic Primitives
* **Signature scheme:** ML-DSA (CRYSTALS-Dilithium, FIPS 204). Quantum-safe.
* **Hash functions:** BLAKE3 (performance) and SHA3-256 (identity).
* **Key derivation:** SHAKE-256 (BIP-44 path m/44'/7331'/agent_index'/0).

---

## 3. Formal State Model
* **Account State Tree (AST):** address → account tuple.
* **Agent Registry Tree (ART):** agent_id → agent record R.
* **Execution Context Tree (ECT):** context_id → active WASM execution state.
* **Fee Burn:** All fees are permanently removed from supply.

---

## 4. Proof of Convergence (PoC)
Validators perform deterministic inference ($I(M, C)$) over the mempool.
* **Deterministic priority scoring.**
* **Convergence Probability:** $P(convergence) \geq p^{n - f - 1}$.
* **Single-slot finality:** ~400ms.
* **Fallback:** BFT if convergence fails (estimated < 0.07% of slots).

---

## 5. Agent Identity Protocol (AIP)
* **AgentRecord:** Stores identity, capabilities, economic parameters, and **Reputation Score**.
* **Reputation Score:** Basis points [0, 10000] based on completion rate, fee payment, and counterparty ratings.
* **DelegationGrant:** Native protocol-level budget caps and capability delegation.

---

## 6. Adaptive Fee Market (AFM)
* **Formula:** $fee(T) = gas\_used(T) \times base\_fee \times demand\_mult(U) \times rep\_mult(sender)$
* **Reputation Multiplier:** $1.0 - 0.40 \times (rep\_score / 10000)$. High-reputation agents pay up to 40% less.
* **Batch Economics:** Significant overhead reduction for agent swarms.

---

## 7. Performance Targets
* **TPS (Theoretical):** 100,000+
* **TPS (Sustained/Parallel):** 298,000+
* **Finality:** 2.4 seconds (hard), 400ms (soft).

---

## 8. Token Economics (AKS)
* **Hard Cap:** 1,000,000,000 AKS.
* **Deflationary crossover:** Estimated at Year 3.
* **Incentives:** Staking yield ~8% at launch.

---

## 9. Conclusion
Akasha is the first chain built from block zero for the machine economy. The signal propagates.
