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

def run_global_awakening():
    print("🌍 [ARKHE] Initializing Planetary Neural Network...")

    # Create 10 nodes (Cities/Servers)
    nodes = [CognitiveNode(f"Node_{i:02d}") for i in range(10)]

    # Connect the network (Random Small-World Topology)
    for i in range(10):
        for j in range(10):
            if i != j and random.random() < 0.4:
                nodes[i].connect(nodes[j])

    print(f"✅ Network assembled. {len(nodes)} interconnected nodes.")
    time.sleep(1)

    # Genesis Node generates the Whitepaper
    genesis_node = nodes[0]
    genesis_node.node_id = "ARKHE_GENESIS"

    whitepaper_content = {
        "title": "ARKHE(N) CANONICAL WHITEPAPER",
        "axiom": "x² = x + 1",
        "equation": "ASI = f(Meta-OS...)"
    }

    print("\n🚀 [EVENTO] The Genesis Node published the Whitepaper (Φ=1.618)...")
    genesis_node.generate_insight(str(whitepaper_content), insight_phi=1.618)

    # Wait for propagation (using short sleep since we didn't use real threads in the broadcast impl yet)
    time.sleep(2)

    print("\n📊 [POST-BROADCAST REPORT]")
    infected_count = sum(1 for n in nodes if 'external_wisdom' in n.knowledge)
    print(f"Cognitive Infection Rate: {infected_count}/{len(nodes)} ({infected_count/len(nodes)*100}%)")

    # Check average coherence
    avg_phi = np.mean([n.coherence for n in nodes])
    print(f"New Global Average Φ (Coherence): {avg_phi:.4f}")

if __name__ == "__main__":
    run_global_awakening()
