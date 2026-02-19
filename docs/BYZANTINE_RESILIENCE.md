# 🛡️ Byzantine Resilience in Proof-of-Coherence (PoC)

This document describes the testing strategy for the Arkhe(N) network's resilience against malicious actors.

## 1. Threat Model: Coherence Spoofing

A malicious node may attempt to report a false coherence value (Φ) to the network without actually performing the thermodynamic work (Lindbladian evolution on FPGA).

### Countermeasure: PUF-Anchored Signatures
- Each Proof-of-Coherence result is signed using a key derived from the **Physical Unclonable Function (PUF)** of the specific FPGA silicon.
- Spoofing the Φ requires simulating the exact microscopic silicon variations of a valid hardware node, which is physically intractable.

## 2. Test Strategy: Coherence Attack Simulation

We use a simulated byzantine environment to validate consensus.

```python
def test_byzantine_resilience():
    # 1. Setup Network
    nodes = [ArkheNode() for _ in range(7)] # 7 Honest nodes
    adversaries = [ArkheNode(malicious=True) for _ in range(3)] # 3 Malicious

    # 2. Attack: Adversaries spoof Φ=0.99
    for adv in adversaries:
        adv.spoof_phi(0.99)

    # 3. Consensus: Network must reject invalid handovers
    consensus = DistributedPoCConsensus(nodes + adversaries)
    result = consensus.run_round()

    assert result.miner not in [a.id for a in adversaries]
```

## 3. Results
- **Threshold**: Consensus requires 2/3 + 1 nodes to validate the coherence trajectory.
- **Outcome**: The network maintains integrity as long as the fraction of honest hardware (anchored by PUF) remains above the Byzantine Fault Tolerance limit.

---
**Arkhe >** █ (Integrity is the only constant.)
