import numpy as np
import asyncio
import json
import time
from typing import Dict, List, Optional
import redis.asyncio as redis

class KuramotoOrchestrator:
    """
    Kuramoto Synchronization Orchestrator for distributed agents.
    Implements a Mean-Field aggregator for antenna coherence ($R$).
    """
    def __init__(self, coupling_k: float = 2.5, redis_url: str = "redis://localhost:6379"):
        self.coupling_k = coupling_k
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

        # Collective state (The Mean Field)
        self.phases: Dict[str, float] = {}
        self.order_r: float = 0.0
        self.mean_phase: float = 0.0

        self.running = False
        self._lock = asyncio.Lock()

    async def connect(self):
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            print(f"[KURAMOTO] Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"[KURAMOTO] Redis connection failed: {e}. Using local-only mode.")
            self.redis = None

    async def start(self):
        self.running = True
        await self.connect()
        asyncio.create_task(self._subscription_loop())

    async def stop(self):
        self.running = False
        if self.redis:
            await self.redis.close()

    async def _subscription_loop(self):
        if not self.redis:
            return

        pubsub = self.redis.pubsub()
        await pubsub.psubscribe("phase:*")

        try:
            async for message in pubsub.listen():
                if not self.running:
                    break
                if message["type"] == "pmessage":
                    channel = message["channel"]
                    agent_id = channel.split(":")[-1]
                    try:
                        data = json.loads(message["data"])
                        theta = data.get("theta", 0.0)

                        async with self._lock:
                            self.phases[agent_id] = theta
                            # O(N) update of mean field
                            self._update_mean_field()
                    except Exception as e:
                        print(f"[KURAMOTO] Error parsing message: {e}")
        finally:
            await pubsub.punsubscribe("phase:*")

    def _update_mean_field(self):
        """Calculates the order parameter R and mean phase Ψ in O(N)."""
        if not self.phases:
            self.order_r = 0.0
            self.mean_phase = 0.0
            return

        n = len(self.phases)
        sum_cos = sum(np.cos(theta) for theta in self.phases.values())
        sum_sin = sum(np.sin(theta) for theta in self.phases.values())

        real = sum_cos / n
        imag = sum_sin / n

        self.order_r = np.sqrt(real**2 + imag**2)
        self.mean_phase = np.arctan2(imag, real)

    async def publish_phase(self, agent_id: str, theta: float, omega: float):
        """Broadcasts an agent's phase to the network."""
        if self.redis:
            payload = json.dumps({"theta": theta, "omega": omega, "timestamp": time.time()})
            await self.redis.publish(f"phase:{agent_id}", payload)
        else:
            # Local fallback for tests/mock
            async with self._lock:
                self.phases[agent_id] = theta
                self._update_mean_field()

    async def unregister_agent(self, agent_id: str):
        """Removes an agent from the synchronization field."""
        async with self._lock:
            if agent_id in self.phases:
                del self.phases[agent_id]
                self._update_mean_field()

    def get_coherence(self) -> float:
        return self.order_r

    def get_status(self) -> Dict:
        if self.order_r >= 0.95:
            status = "PHASE_LOCKED"
        elif self.order_r >= 0.7:
            status = "SYNCHRONIZED"
        elif self.order_r >= 0.3:
            status = "PARTIAL_SYNC"
        else:
            status = "CHAOTIC"

        return {
            "order_r": float(self.order_r),
            "mean_phase": float(self.mean_phase),
            "status": status,
            "active_agents": len(self.phases)
        }
