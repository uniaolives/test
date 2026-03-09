import asyncio
import numpy as np
import os
import sys
import time

# Add the project root to sys.path to import the gateway modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.app.physics.kuramoto import KuramotoOrchestrator

async def test_spatial_anomaly_detection():
    print("Starting Spatial Anomaly Detection Test (Orb Simulation)...")

    # Initialize orchestrator
    orchestrator = KuramotoOrchestrator(coupling_k=5.0, redis_url="redis://localhost:6379")
    orchestrator.redis = None # Local fallback
    orchestrator.running = True

    # 1. Setup global background agents (Low coherence)
    n_global = 20
    for i in range(n_global):
        await orchestrator.publish_phase(
            f"global_{i}",
            np.random.uniform(0, 2 * np.pi), 1.0,
            lat=0.0, lon=0.0, altitude=0.0, phi_q=1.0
        )

    # 2. Setup a local cluster peaking in coherence (The Orb)
    # Location: Rio de Janeiro (-22.9, -43.1)
    # φ_q > 4.64 (Miller Limit)
    n_local = 5
    for i in range(n_local):
        await orchestrator.publish_phase(
            f"rio_node_{i}",
            1.0, 1.0, # Synchronized local phases
            lat=-22.9068, lon=-43.1729, altitude=15.0, phi_q=9.5 # High phi_q
        )

    print(f"Simulating {n_global + n_local} agents...")

    # Trigger detection
    await orchestrator.detect_spatial_anomalies()

    status = orchestrator.get_status()
    print(f"Global Coherence R: {status['order_r']:.4f}")
    print(f"Anomalies Detected: {status['detected_anomalies']}")

    if status['detected_anomalies'] > 0:
        anomaly = orchestrator.anomalies[0]
        print(f"SUCCESS: Spatial Anomaly (Orb) detected at {anomaly['lat']}, {anomaly['lon']}")
        print(f"Orb Classification: {anomaly['classification']}")
        print(f"Orb Intensity: {anomaly['intensity']:.4f}")
    else:
        print("FAILURE: System failed to detect the simulated Orb.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_spatial_anomaly_detection())
