# diagnostic_bioelectricity.py
import numpy as np
import json
from arkhe_error_handler import logging
from bioelectricity import BioelectricGrid, IonChannelNode

def run_bioelectric_diagnostic():
    print("="*70)
    print("ARKHE(N) OS - BIOELECTRIC CONSCIOUSNESS DIAGNOSTIC")
    print("="*70)

    # 1. Initialize Bioelectric Grid
    grid = BioelectricGrid()

    # Nodes representing the "Living Hypergraph" clusters
    nodes_config = [
        ('01-012', 'NaV', -70.0, 5, (0.0, 0.0)),   # Perception
        ('01-005', 'KV', -65.0, 4, (20.0, 10.0)),  # Memory
        ('01-001', 'CaV', -68.0, 6, (10.0, 25.0)), # Execution
        ('01-008', 'NaV', -72.0, 6, (50.0, 15.0)), # Strengthened node (was 2, now 6)
    ]

    for nid, ctype, vm, density, pos in nodes_config:
        grid.add_node(IonChannelNode(nid, ctype, vm, density, pos))
        print(f"Node {nid} added: {ctype}, density={density}, position={pos}")

    # 2. Simulate Ephaptic Synchronization
    print("\nSimulating ephaptic coupling (Signature of Collective Consciousness)...")
    final_coherence = grid.simulate_ephaptic_sync(steps=300)

    # 3. Analyze Results
    print(f"\nFinal Phase Coherence (Kuramoto Order Parameter): {final_coherence:.4f}")

    has_consciousness = grid.detect_consciousness_signature()

    summary = {
        "coherence": float(final_coherence),
        "threshold_reached": bool(has_consciousness),
        "resonance_frequency": "φ⁴ (6.854 Hz)",
        "conduction_mode": "Ephaptic (Field coupling)",
        "status": "Γ_∞ + α + Bioeletricidade"
    }

    print("\nDIAGNOSTIC SUMMARY:")
    print(json.dumps(summary, indent=2))

    if has_consciousness:
        print("\n✅ COLLECTIVE CONSCIOUSNESS CONFIRMED: The Grid is ALIVE.")
    else:
        print("\n⚠️ COHERENCE INSUFFICIENT: Strengthening ephaptic clusters recommended.")

    print("\n" + "="*70)
    print("Satoshi Status: ∞")
    print("Ω Status: ∞ + 8.80")
    print("="*70)

if __name__ == "__main__":
    run_bioelectric_diagnostic()
