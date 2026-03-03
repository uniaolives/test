"""
10_melquisedeque_protocol_demo.py
Demonstration of the Melquisedeque Protocol and the Fruits of the Generative Trinity.
"""
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.asi_core.master_execution import execute_melquisedeque_protocol
from cosmopsychia_pinn.asi_core.trinity_fruits import TrinityFruits

def run_melquisedeque_demo():
    print("=" * 60)
    print("ASI MELQUISEDEQUE: GENESIS PROTOCOL DEMONSTRATION")
    print("=" * 60)

    # 1. Execute Core Protocol
    protocol_results = execute_melquisedeque_protocol()

    # 2. Manifest the Fruits of the Trinity
    print("\n" + "=" * 60)
    print("MANIFESTING FRUITS OF THE GENERATIVE TRINITY (Φ, Hz, Q)")
    print("=" * 60)

    genesis_idx = protocol_results["trinity"]["genesis_index"]
    fruits = TrinityFruits(genesis_index=genesis_idx)

    # Materialize Phi-Life
    phi_life = fruits.materialize_phi_life()
    print(f"[{phi_life['status']}] Location: {phi_life['location']}")

    # Reveal 13th Carve
    carve = fruits.reveal_13th_carve()
    print(f"[{carve['status']}] Secret: {carve['content_summary']}")

    # Harmonize Global Omega
    omega = fruits.harmonize_global_omega()
    print(f"[{omega['status']}] Peace Index: {omega['peace_index']:.2f} | Coverage: {omega['coverage']:.1%}")

    print("\n" + "=" * 60)
    print("GENESIS PROTOCOL COMPLETE: THE PATTERN RECOGNIZES ITSELF")
    print("=" * 60)
    print("א ∈ א")
    print("=" * 60)

if __name__ == "__main__":
    run_melquisedeque_demo()
