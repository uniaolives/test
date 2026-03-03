# scripts/verify_meta_handover.py
import sys
import os
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

# Load protocol and meta-observability
protocol_path = root / "arscontexta" / ".arkhe" / "handover" / "protocol.py"
protocol_module = utils.load_arkhe_module(protocol_path, "arkhe.handover.protocol")
MetaHandover = protocol_module.MetaHandover
ArkheNode = protocol_module.ArkheNode
meta_obs = protocol_module.meta_obs

from src.papercoder_kernel.linux.arkhe_driver_sim import ArkheDriverSim

def run_verification():
    print("--- Arkhe(N) Meta-Operational & Meta-Observability Verification ---")

    # 1. Setup Nodes
    linux_driver = ArkheDriverSim("linux-node-01")
    linux_coherence = 0.85

    linux_node = ArkheNode("linux-node-01", coherence=linux_coherence)
    eth_node = ArkheNode("ethereum-node", coherence=0.96)

    print(f"[STATUS] Initial Linux Node Coherence: {linux_node.coherence:.4f}")
    print(f"[STATUS] Initial Ethereum Node Coherence: {eth_node.coherence:.4f}")

    # 2. Simulate multiple handovers
    print("\n[INFO] Executing multiple handovers to activate Meta-Observability...")
    for i in range(25):
        handover_req = linux_driver.syscall_to_handover(f"op_{i}", (i,))
        handover = MetaHandover(
            source_id=handover_req["source_id"],
            target_id="ethereum-node",
            payload=handover_req["payload"],
            coherence_in=handover_req["coherence_in"],
            coherence_out=0.90,
            phi_required=0.001
        )
        success = handover.validate_and_execute(linux_node, eth_node, global_phi=0.005)
        if not success:
            break

    # 3. Check Meta-Observability Status
    print("\n--- Meta-Observability Report ---")
    report = meta_obs.get_status_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # 4. Check for Metamorphosis
    decision = meta_obs.should_metamorphose()
    print(f"\n[DECISION] Meta-Operational Metamorphosis: {decision}")

    if success:
        print("\n[SUCCESS] Functional verification complete.")
    else:
        print("\n[FAILURE] Functional verification failed.")
        sys.exit(1)

    linux_driver.cleanup()

if __name__ == "__main__":
    run_verification()
