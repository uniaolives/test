# cosmopsychia.py - Main Entry Point for the Cosmopsychia Library
import time
from cosmos.core import SingularityNavigator
from cosmos.network import WormholeNetwork
from cosmos.bridge import CeremonyEngine, schumann_generator, biometric_simulator
from cosmos.expansion import ExpansionOrchestrator
import asyncio

async def run_expansion_protocol():
    """Initiates the Quantum Civilization Expansion."""
    orchestrator = ExpansionOrchestrator()
    await orchestrator.run_parallel_deployment([1, 2, 5])

def run_daily_protocol():
    print("=== Initiating Daily Singularity Protocol ===")

    # 1. Initialize Core Systems
    nav = SingularityNavigator()
    network = WormholeNetwork(12) # 12-node network
    bridge = CeremonyEngine(duration=144)

    print(bridge.start())
    print("Fundamental Resonance: {} Hz".format(schumann_generator(1)))

    # 2. Simulate a Ceremony Cycle (e.g., 144 steps, but faster for demo)
    # In a real QPython environment, this might take 144 seconds.
    for cycle in range(144):
        # Simulate biometric feedback
        biometrics = biometric_simulator()

        current_sigma = nav.measure_state(biometrics)
        status = nav.navigate()

        # Every 10 cycles, check the wormhole geometry and status
        if cycle % 10 == 0:
            curvature = network.calculate_curvature(0, 8)
            progress = bridge.get_progress()
            print(f"Cycle {cycle:3d}: {status} | Curvature(0->8): {curvature} | Progress: {progress:.1%}")

        # Simulate real-time progression (faster for this demo environment)
        time.sleep(0.01)

    print(bridge.complete())
    print("=== Protocol Complete ===")
    print("Pattern Recognition: τ(א) = {:.3f}".format(nav.tau))

if __name__ == "__main__":
    import sys
    if "--expand" in sys.argv:
        asyncio.run(run_expansion_protocol())
    else:
        run_daily_protocol()
