# cosmopsychia.py - Main Entry Point for the Cosmopsychia Library
import time
from cosmos.core import SingularityNavigator
from cosmos.network import WormholeNetwork
from cosmos.bridge import (
    AdvancedCeremonyEngine,
    TimeLockCeremonyEngine,
    visualize_timechain_html,
    schumann_generator
)
from cosmos.ontological import OntologicalKernel
from cosmos.service import CosmopsychiaService

def run_daily_protocol():
    print("=== Initiating Daily Singularity Protocol ===")

    # 1. Initialize Core Systems
    # We use AdvancedCeremonyEngine which bundles Navigator and Network
    base_engine = AdvancedCeremonyEngine(duration=144, node_count=12)

    # 2. Integrate with qTimeChain via TimeLockCeremonyEngine
    time_engine = TimeLockCeremonyEngine(base_engine)

    print("Fundamental Resonance: {} Hz".format(schumann_generator(1)))

    # 3. Execute Time-Locked Ceremony (demonstration for 15 seconds)
    print("\n🚀 Starting Time-Locked Ceremony (Demonstration)...")
    time_engine.execute_time_locked_ceremony(duration_seconds=15)

    # 4. Generate Visualization
    print("\n📊 Generating visualization...")
    viz_result = visualize_timechain_html(
        time_engine.timechain,
        filename="quantum_timechain_viz.html"
    )
    print(viz_result)

    # 5. Ontological and Service Checks
    print("\n🧐 Running Ontological and Service Checks...")
    kernel = OntologicalKernel()
    service = CosmopsychiaService()

    health = service.check_substrate_health()
    print(f"Substrate Health: {health['status']} (Score: {health['health_score']:.2f})")

    oracle = service.get_oracle_insight()
    print(f"Quantum Oracle Insight: {oracle['suggested_emergence']} (QRNG: {oracle['qrng_value']:.4f})")

    print("\n=== Protocol Complete ===")
    latest_tau = time_engine.timechain.chain[-1].ceremony_state.get('tau', 0)
    print("Pattern Recognition: τ(א) = {:.3f}".format(latest_tau))

if __name__ == "__main__":
    run_daily_protocol()
