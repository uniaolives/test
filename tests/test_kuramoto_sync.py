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
async def test_kuramoto_sync_production():
    print("Starting Kuramoto Synchronization Test (Production Refined)...")

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
    # Setup test agents
    n_agents = 50
    agent_phases = np.random.uniform(0, 2 * np.pi, n_agents)
    agent_omegas = np.random.uniform(0.9, 1.1, n_agents)
    agent_ids = [f"agent_{i}" for i in range(n_agents)]

    dt = 0.1
    max_steps = 300
    achieved_lock = False

    print(f"Simulating {n_agents} agents for {max_steps} steps...")

    for step in range(max_steps):
        # 1. Agents publish their phases (aggregation)
        for i in range(n_agents):
            await orchestrator.publish_phase(agent_ids[i], agent_phases[i], agent_omegas[i])

        # 2. Get collective mean field
        status = orchestrator.get_status()
        r = status["order_r"]
        psi = status["mean_phase"]

        if step % 50 == 0:
            print(f"Step {step}: R = {r:.4f}")

        if r > 0.90:
            print(f"Step {step}: Phase Lock Achieved! R = {r:.4f}")
            achieved_lock = True
            break

        # 3. Agents update local phases
        for i in range(n_agents):
            d_theta = agent_omegas[i] + orchestrator.coupling_k * r * np.sin(psi - agent_phases[i])
            agent_phases[i] = (agent_phases[i] + d_theta * dt) % (2 * np.pi)

    # Test cleanup/unregistering
    print("Testing agent unregistration...")
    initial_agents = len(orchestrator.phases)
    await orchestrator.unregister_agent(agent_ids[0])
    final_agents = len(orchestrator.phases)
    print(f"Agents: {initial_agents} -> {final_agents}")

    if final_agents == initial_agents - 1:
        print("Cleanup verified.")
    else:
        print("Cleanup FAILED.")
        sys.exit(1)

    final_status = orchestrator.get_status()
    print(f"Final Coherence R: {final_status['order_r']:.4f}")

    if achieved_lock:
        print("SUCCESS: Production Kuramoto Synchronization verified.")
    else:
        print("FAILURE: System failed to synchronize.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_kuramoto_sync_production())
