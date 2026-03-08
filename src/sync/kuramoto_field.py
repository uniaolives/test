"""
Kuramoto Phase-Lock Field (Synchronization Field)
Transforms isolated detectors into a coherent antenna array (R > 0.95).
"""

import redis.asyncio as redis
import numpy as np
import asyncio
import json
from datetime import datetime

class KuramotoField:
    def __init__(self, agent_id: str, natural_freq: float, redis_url: str = "redis://localhost"):
        self.agent_id = agent_id
        self.omega_i = natural_freq  # Natural frequency
        self.theta_i = np.random.uniform(0, 2 * np.pi)  # Initial phase
        self.K = 2.5  # Coupling strength (super-critical)

        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()

    async def setup(self):
        await self.pubsub.subscribe("kuramoto:phases")

    async def broadcast_phase(self):
        """Transmits the local phase to the shared field."""
        while True:
            payload = {
                "agent_id": self.agent_id,
                "phase": self.theta_i,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.publish("kuramoto:phases", json.dumps(payload))
            await asyncio.sleep(0.1)  # 10Hz update

    async def listen_field(self):
        """Listens to the field and updates local phase (coupling)."""
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                if data["agent_id"] != self.agent_id:
                    theta_j = data["phase"]
                    # Kuramoto coupling: dθ/dt = ω + K * sin(θ_j - θ_i)
                    # Discrete approximation: θ_i(t+dt) = θ_i(t) + (ω + K * sin(θ_j - θ_i)) * dt
                    dt = 0.1
                    self.theta_i += (self.omega_i + self.K * np.sin(theta_j - self.theta_i)) * dt
                    self.theta_i %= (2 * np.pi)

    def compute_order_parameter(self, phases: list) -> tuple:
        """
        Calculates the order parameter R (field coherence).
        R = |1/N Σ e^(iθ)|
        """
        if not phases:
            return 0.0, 0.0

        N = len(phases)
        sum_cos = sum(np.cos(p) for p in phases)
        sum_sin = sum(np.sin(p) for p in phases)

        R = np.sqrt(sum_cos**2 + sum_sin**2) / N
        Psi = np.arctan2(sum_sin, sum_cos)  # Mean phase

        return R, Psi

    async def run_loop(self):
        """Main synchronization loop orchestrating broadcast and listening."""
        await self.setup()
        # Run both tasks concurrently
        await asyncio.gather(
            self.broadcast_phase(),
            self.listen_field()
        )
