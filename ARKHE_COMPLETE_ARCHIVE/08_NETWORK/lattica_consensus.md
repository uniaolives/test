# Lattica Consensus: Coherence-Based Protocol

## Principle
Consensus is not achieved by majority vote, but by convergence of syzygy (⟨ϕ₁|ϕ₂⟩) across the network.

## Implementation
1.  **Node Identification**: Nodes are identified by their Larmor frequency (ν).
2.  **Handover Validation**: A handover is considered "canonized" if the resulting syzygy is > 0.94.
3.  **Conflict Resolution**: In case of divergence, the path with the highest integrated coherence (∫C dt) is selected as the canonical geodesic.
4.  **Proof of Coherence**: Nodes must prove they maintain C ≈ 0.86 and F ≈ 0.14 to participate in the high-integrity backbone.

## API
```typescript
interface HandoverProposal {
  source_id: string;
  target_id: string;
  payload_hash: string;
  phi_signature: number;
}
```
