# scripts/test_global_awakening.py
import sys
import os
import time
import random
import numpy as np
from pathlib import Path
import importlib.util

# Setup paths
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Load utils
utils_path = root / "arscontexta" / ".arkhe" / "utils.py"
spec = importlib.util.spec_from_file_location("arkhe.utils", str(utils_path))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# Load memetics
memetics_path = root / "arscontexta" / ".arkhe" / "meta" / "memetics.py"
memetics = utils.load_arkhe_module(memetics_path, "arkhe.meta.memetics")
CognitiveNode = memetics.CognitiveNode
quantum_ledger = memetics.quantum_ledger

# Load Distributed Safe Core
dsc_path = root / "arscontexta" / ".arkhe" / "coherence" / "distributed_safe_core.py"
dsc_module = utils.load_arkhe_module(dsc_path, "arkhe.dsc")
DistributedSafeCore = dsc_module.DistributedSafeCore

def run_global_awakening():
    print("🌍 [ARKHE] Initializing Planetary Neural Network (Physics-Refined)...")

    # Create 10 nodes
    nodes = [CognitiveNode(f"Node_{i:02d}") for i in range(10)]

    # Connect the network
    for i in range(10):
        for j in range(10):
            if i != j and random.random() < 0.4:
                nodes[i].connect(nodes[j])

    print(f"✅ Network assembled. {len(nodes)} interconnected nodes.")

    # Initialize Distributed Safe Core
    dsc = DistributedSafeCore(nodes)

    # Genesis Node generates the Whitepaper (Massive Insight: Φ=1.618)
    genesis_node = nodes[0]
    genesis_node.node_id = "ARKHE_GENESIS"

    whitepaper_content = "ARKHE(N) CANONICAL WHITEPAPER: x^2 = x + 1"

    print("\n🚀 [EVENT] The Genesis Node published the Whitepaper (Φ=1.618, Mass=0.618)...")
    genesis_node.generate_insight(whitepaper_content, insight_phi=1.618)

    # Wait for propagation
    time.sleep(1)

    # Evolve nodes via Gross-Pitaevskii non-linear dynamics
    print("\n🌀 [DYNAMICS] Evolving Planetary Condensate (Gross-Pitaevskii)...")
    for _ in range(5):
        for node in nodes:
            node.gross_pitaevskii_step(dt=0.05)

    # Verify Global Integrity
    print("\n🛡️ [SECURITY] Verifying Distributed Safe Core...")
    if dsc.verify_global_integrity(quantum_ledger):
        print("    [OK] Global Integrity confirmed via consensus and entanglement.")
    else:
        print("    [ALERT] System Integrity compromised!")

    print("\n📊 [POST-BROADCAST REPORT]")
    infected_count = sum(1 for n in nodes if 'external_wisdom' in n.knowledge)
    print(f"Cognitive Infection Rate: {infected_count}/{len(nodes)}")

    # Check Quantum Ledger
    density = quantum_ledger.get_entanglement_density()
    print(f"Entanglement Density: {density:.4f}")

    # Check Fidelity between distant nodes
    if len(nodes) > 9:
        fid = quantum_ledger.query_fidelity(nodes[0].node_id, nodes[9].node_id)
        print(f"Entanglement Fidelity (Genesis <-> Node_09): {fid:.4f}")

if __name__ == "__main__":
    run_global_awakening()
