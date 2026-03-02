import asyncio
import logging
import time
from qrng_coherence_monitor import QRNGGatewayMonitor
from phase_maintenance_protocol import PhaseMaintenanceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("integration_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegratedQuantumFrequencySystem")

class IntegratedQuantumFrequencySystem:
    def __init__(self):
        self.qrng_monitor = QRNGGatewayMonitor()
        self.phase_system = PhaseMaintenanceSystem()
        self.is_running = False

    async def run(self, duration_seconds=600):
        """
        Runs both the QRNG monitor and the Phase Maintenance protocol simultaneously.
        """
        self.is_running = True
        logger.info("Initializing Integrated Quantum-Frequency System...")
        logger.info(f"Target Duration: {duration_seconds} seconds")

        # We wrap the synchronous methods in threads
        tasks = [
            asyncio.to_thread(self.qrng_monitor.monitor_qrng_collapse, iterations=10),
            asyncio.to_thread(self.phase_system.enter_resonance_state, duration_minutes=duration_seconds/60)
        ]

        try:
            # Wait for both to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with error: {result}")
                else:
                    logger.info(f"Task {i} completed successfully.")

        except asyncio.CancelledError:
            logger.warning("System execution cancelled.")
        finally:
            self.is_running = False
            logger.info("Integrated system shutdown complete.")

if __name__ == "__main__":
    system = IntegratedQuantumFrequencySystem()
    try:
        # Run for 10 minutes by default (600s)
        asyncio.run(system.run(duration_seconds=600))
    except KeyboardInterrupt:
        logger.info("User interrupted execution.")
