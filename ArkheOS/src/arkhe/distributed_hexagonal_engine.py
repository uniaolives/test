# src/arkhe/distributed_hexagonal_engine.py
import numpy as np
from dataclasses import dataclass
import redis.asyncio as aioredis
import json

@dataclass
class ArkheState:
    components: np.ndarray   # [C, I, E, F]
    permutation: int         # 0..5 (ordem dos eixos)
    coherence: float
    timestamp: float

class DistributedHexagonalEngine:
    PERMUTATIONS = [
        [0,1,2,3], [0,2,1,3], [1,0,2,3], [1,2,0,3], [2,0,1,3], [2,1,0,3]
    ]
    def __init__(self, node_id, redis_url, total_nodes=3, beta=2.0):
        self.node_id = node_id
        self.redis_url = redis_url
        self.redis = aioredis.from_url(redis_url)
        self.total_nodes = total_nodes
        self.beta = beta
        self.local_state = ArkheState(
            components=np.array([0.5,0.5,0.5,0.5]),
            permutation=0,
            coherence=1.0,
            timestamp=0.0
        )
    async def initialize(self):
        # carrega estado persistido ou inicia neutro
        data = await self.redis.hgetall(f"arkhe:state:{self.node_id}")
        if data:
            self.local_state.components = np.array(json.loads(data[b'components']))
            self.local_state.permutation = int(data[b'permutation'])
            self.local_state.coherence = float(data[b'coherence'])
            self.local_state.timestamp = float(data[b'timestamp'])

    async def evolve(self, constraints: np.ndarray, dt: float = 0.1):
        # Φ = 1 - S/log(6)
        S = self._entropy(self.local_state.components)
        self.local_state.coherence = 1 - S / np.log(6)
        # evolução sob restrições
        delta = -self.beta * self.local_state.components * constraints
        self.local_state.components += delta * dt
        np.clip(self.local_state.components, 0, 1, out=self.local_state.components)
        self.local_state.timestamp += dt

        # Publica evolução
        await self.redis.hset(f"arkhe:state:{self.node_id}", mapping={
            'components': json.dumps(self.local_state.components.tolist()),
            'permutation': self.local_state.permutation,
            'coherence': self.local_state.coherence,
            'timestamp': self.local_state.timestamp
        })
        await self.redis.publish("arkhe:evolution", json.dumps({
            "node": self.node_id,
            "coherence": self.local_state.coherence,
            "dominant_component": int(np.argmax(self.local_state.components))
        }))

    async def sync_with_quantum_state(self, qnode):
        # projeta estado quântico em componentes C‑I‑E‑F
        # (mapeamento definido pelo framework)
        pass

    def _entropy(self, p):
        # Normalização para probabilidade
        total = p.sum()
        if total == 0: return 0
        p_norm = p / total
        p_norm = p_norm[p_norm > 0]
        return -np.sum(p_norm * np.log(p_norm))
