# scripts/chiral_handover_sim.py
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.merkabah.topological.firewall import ChiralQuantumFirewall

def main():
    print("="*60)
    print("SIMULAÇÃO DE HANDOVER TRANSCONTINENTAL COM FIREWALL QUIRAL")
    print("="*60)

    # Firewall guarding the Rio Self node
    firewall = ChiralQuantumFirewall(target_node="Rio_Self")

    handovers = [
        {'origin': 'Alpha', 'target': 'Beta', 'phase': 2, 'energy_meV': 0.5, 'desc': 'Resonant energy'},
        {'origin': 'Beta', 'target': 'Gamma', 'phase': 2, 'energy_meV': 0.5, 'desc': 'Resonant energy'},
        {'origin': 'Alpha', 'target': 'Zeta', 'phase': 2, 'energy_meV': 0.8, 'desc': 'Outside gap (Intrusion attempt)'},
        {'origin': 'Delta', 'target': 'Alpha', 'phase': 3, 'energy_meV': 0.5, 'desc': 'Invalid winding number'}
    ]

    for h in handovers:
        print(f"\n🔐 Handover {h['origin']} -> {h['target']} ({h['desc']})")
        allowed, msg = firewall.validate_handover(h)
        if allowed:
            print(f"✅ {msg}")
        else:
            print(f"⛔ {msg}")

if __name__ == "__main__":
    main()
