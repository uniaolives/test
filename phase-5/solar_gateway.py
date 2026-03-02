import asyncio
import logging
import sys
import os
from datetime import datetime

# Ensure the current directory is in the path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from solar_gateway_execution import SolarGatewayProtocol
    from integrated_quantum_frequency_system import IntegratedQuantumFrequencySystem
    from wormhole_manifesto import WormholeNetwork
    from cosmopsychia import SingularityNavigator, AdvancedWormholeNetwork, AdvancedCeremonyEngine
except ImportError as e:
    print(f"Import Error: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CURRENT_DIR, "solar_gateway_master.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SolarGatewayMaster")

class GatewayRunner:
    def __init__(self):
        self.protocol = SolarGatewayProtocol(datetime.now().isoformat())
        self.navigator = SingularityNavigator()
        self.ceremony = AdvancedCeremonyEngine()

    async def execute_sequence(self):
        logger.info("Starting Solar Gateway Sync Sequence...")
        logger.info(self.ceremony.start_ceremony())

        schedule = self.protocol.generate_execution_schedule()
        for cell in schedule[:3]: # Execute first 3 cells for demo
            await asyncio.to_thread(self.protocol.execute_cell, cell)

            # Integrate Cosmopsychia navigation during sync
            self.navigator.measure_state(cell)
            nav_result = self.navigator.navigate()
            logger.info(f"Navigation Status: {nav_result['status']} | σ: {nav_result['sigma']:.3f} | τ: {nav_result['tau']:.3f}")

        logger.info("Solar Gateway Sync Sequence Complete.")

async def main():
    logger.info("--- STARTING SOLAR GATEWAY MASTER ORCHESTRATOR ---")
    logger.info("Phase 5: Universal Singularity & Stellar Seeding")

    runner = GatewayRunner()
    quantum_system = IntegratedQuantumFrequencySystem()
    wormhole_net = WormholeNetwork()

    duration = 30 # Verification run

    logger.info(f"[Step 1] Launching Parallel Substrates (Duration: {duration}s)...")

    tasks = [
        asyncio.create_task(runner.execute_sequence()),
        asyncio.create_task(quantum_system.run(duration_seconds=duration)),
        asyncio.to_thread(wormhole_net.execute_wormhole_experiment)
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during gateway execution: {e}")

    logger.info("[Step 2] Solar Gateway Master Sequence Complete.")
    logger.info(f"Final Pattern Recognition: τ(א) = {runner.navigator.tau:.3f}")
    logger.info("--- SOLAR GATEWAY MASTER ORCHESTRATOR SHUTDOWN ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
