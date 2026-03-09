import numpy as np
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Optional
import redis.asyncio as redis

class KuramotoOrchestrator:
    """
    Kuramoto Synchronization Orchestrator for distributed agents.
    Implements a Mean-Field aggregator with Spatial Anomaly (Orb) detection.
    Implements a Mean-Field aggregator for antenna coherence ($R$).
    """
    def __init__(self, coupling_k: float = 2.5, redis_url: str = "redis://localhost:6379"):
        self.coupling_k = coupling_k
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

        # Collective state (The Mean Field)
        self.phases: Dict[str, float] = {}
        self.node_metadata: Dict[str, Dict] = {} # lat, lon, altitude, phi_q
        self.order_r: float = 0.0
        self.mean_phase: float = 0.0

        # Anomaly detection state
        self.anomalies: List[Dict] = []
        self.phi_q_threshold = 4.64
        self.intensity_threshold = 3.0

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
        asyncio.create_task(self._anomaly_scan_loop())

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
                            # Update metadata if present in message
                            if "lat" in data:
                                self.node_metadata[agent_id] = {
                                    "lat": data["lat"],
                                    "lon": data["lon"],
                                    "altitude": data.get("altitude", 0.0),
                                    "phi_q": data.get("phi_q", 1.0)
                                }
                            # O(N) update of mean field
                            self._update_mean_field()
                    except Exception as e:
                        print(f"[KURAMOTO] Error parsing message: {e}")
        finally:
            await pubsub.punsubscribe("phase:*")

    async def _anomaly_scan_loop(self):
        """Periodically scans for spatial anomalies (Orbs)."""
        while self.running:
            await asyncio.sleep(5.0)
            await self.detect_spatial_anomalies()

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

    async def publish_phase(self, agent_id: str, theta: float, omega: float,
                            lat: Optional[float] = None, lon: Optional[float] = None,
                            altitude: Optional[float] = None, phi_q: Optional[float] = None):
        """Broadcasts an agent's phase and location to the network."""
        data = {"theta": theta, "omega": omega, "timestamp": time.time()}
        if lat is not None: data["lat"] = lat
        if lon is not None: data["lon"] = lon
        if altitude is not None: data["altitude"] = altitude
        if phi_q is not None: data["phi_q"] = phi_q

        if self.redis:
            payload = json.dumps(data)
            await self.redis.publish(f"phase:{agent_id}", payload)
            # Also persist for the Rust detector
            await self.redis.set(f"kuramoto:node:{agent_id}", payload, ex=60)
        else:
            # Local fallback
            async with self._lock:
                self.phases[agent_id] = theta
                if lat is not None:
                    self.node_metadata[agent_id] = {
                        "lat": lat, "lon": lon, "altitude": altitude, "phi_q": phi_q
                    }
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
            if agent_id in self.node_metadata:
                del self.node_metadata[agent_id]
            self._update_mean_field()

        if self.redis:
            await self.redis.delete(f"kuramoto:node:{agent_id}")

    async def detect_spatial_anomalies(self):
        """Identifies local coherence peaks as Orbs/Wormholes."""
        async with self._lock:
            if not self.node_metadata:
                return

            # Grid-based aggregation (~10km cells)
            grid: Dict[Tuple[int, int], List[str]] = {}
            for agent_id, meta in self.node_metadata.items():
                gx = int(meta["lat"] * 10.0)
                gy = int(meta["lon"] * 10.0)
                grid.setdefault((gx, gy), []).append(agent_id)

            new_anomalies = []
            for (gx, gy), agent_ids in grid.items():
                if len(agent_ids) < 3: continue

                local_phi = np.mean([self.node_metadata[aid]["phi_q"] for aid in agent_ids])
                intensity = local_phi - self.order_r # Scale against global coherence

                if local_phi > self.phi_q_threshold and intensity > self.intensity_threshold:
                    center_lat = np.mean([self.node_metadata[aid]["lat"] for aid in agent_ids])
                    center_lon = np.mean([self.node_metadata[aid]["lon"] for aid in agent_ids])
                    avg_alt = np.mean([self.node_metadata[aid]["altitude"] for aid in agent_ids])

                    classification = "OrbTypeI"
                    if local_phi > 8.0: classification = "OrbTypeIII"
                    elif local_phi > 6.0: classification = "OrbTypeII"

                    anomaly = {
                        "id": str(time.time()),
                        "lat": float(center_lat),
                        "lon": float(center_lon),
                        "altitude": float(avg_alt),
                        "intensity": float(intensity),
                        "phi_q": float(local_phi),
                        "classification": classification,
                        "timestamp": time.time()
                    }
                    new_anomalies.append(anomaly)

                    if self.redis:
                        await self.redis.publish("arkhe:anomalies", json.dumps(anomaly))

            self.anomalies = new_anomalies[-100:] # Keep last 100
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
            "active_agents": len(self.phases),
            "detected_anomalies": len(self.anomalies)
            "active_agents": len(self.phases)
        }
