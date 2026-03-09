import asyncio
import numpy as np
import os
import sys

# Add the project root to sys.path to import the gateway modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.app.physics.kuramoto import KuramotoOrchestrator

async def test_kuramoto_sync_production():
    print("Starting Kuramoto Synchronization Test (Production Refined)...")

    # Initialize orchestrator
    orchestrator = KuramotoOrchestrator(coupling_k=5.0, redis_url="redis://localhost:6379")
    orchestrator.redis = None # Local fallback
    orchestrator.running = True

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
