"""
Parallax Entanglement Registry (Redis)
Gerencia pares emaranhados globalmente no cluster.
"""

import json
import time
from typing import Optional, Tuple, List, Dict
import redis.asyncio as redis


class EntanglementRegistry:
    """Registro global de pares emaranhados, armazenado no Redis."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "entanglement:"

    async def create_pair(self, node_a: str, agent_a: int,
                          node_b: str, agent_b: int,
                          bell_type: int = 0) -> bool:
        """
        Registra um par emaranhado entre dois agentes em nós distintos.
        Retorna True se criado com sucesso, False se conflito.
        """
        # Garante que nenhum dos agentes já está emaranhado
        if await self._is_entangled(node_a, agent_a) or \
           await self._is_entangled(node_b, agent_b):
            return False

        pair_id = f"{node_a}:{agent_a}:{node_b}:{agent_b}:{int(time.time())}"
        key = f"{self.key_prefix}{pair_id}"

        data = {
            'node_a': node_a,
            'agent_a': agent_a,
            'node_b': node_b,
            'agent_b': agent_b,
            'bell_type': bell_type,
            'created_at': time.time(),
            'active': 1
        }

        mapping = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                   for k, v in data.items()}

        await self.redis.hset(key, mapping=mapping)
        await self.redis.expire(key, 3600)  # expira em 1h

        # Índices para consulta rápida
        await self.redis.sadd(f"{self.key_prefix}node:{node_a}:agents", agent_a)
        await self.redis.sadd(f"{self.key_prefix}node:{node_b}:agents", agent_b)
        await self.redis.set(f"{self.key_prefix}agent:{node_a}:{agent_a}", key, ex=3600)
        await self.redis.set(f"{self.key_prefix}agent:{node_b}:{agent_b}", key, ex=3600)

        return True

    async def get_partner(self, node: str, agent: int) -> Optional[Tuple[str, int, int]]:
        """
        Retorna (nó_parceiro, id_agente_parceiro, bell_type) ou None.
        """
        key = await self.redis.get(f"{self.key_prefix}agent:{node}:{agent}")
        if not key:
            return None

        data = await self.redis.hgetall(key)
        if not data or data.get('active') != '1':
            return None

        node_a = data.get('node_a', '')
        agent_a = int(data.get('agent_a', 0))
        node_b = data.get('node_b', '')
        agent_b = int(data.get('agent_b', 0))
        bell_type = int(data.get('bell_type', 0))

        if node == node_a and agent == agent_a:
            return node_b, agent_b, bell_type
        elif node == node_b and agent == agent_b:
            return node_a, agent_a, bell_type
        else:
            return None

    async def break_pair(self, node: str, agent: int) -> bool:
        """Remove um par emaranhado (após colapso)."""
        key = await self.redis.get(f"{self.key_prefix}agent:{node}:{agent}")
        if not key:
            return False

        # Obtém dados antes de remover índices
        data = await self.redis.hgetall(key)
        if data:
            node_a = data.get('node_a', '')
            agent_a = int(data.get('agent_a', 0))
            node_b = data.get('node_b', '')
            agent_b = int(data.get('agent_b', 0))

            await self.redis.srem(f"{self.key_prefix}node:{node_a}:agents", agent_a)
            await self.redis.srem(f"{self.key_prefix}node:{node_b}:agents", agent_b)
            await self.redis.delete(f"{self.key_prefix}agent:{node_a}:{agent_a}")
            await self.redis.delete(f"{self.key_prefix}agent:{node_b}:{agent_b}")

            # Marca como inativo no hash original
            await self.redis.hset(key, 'active', '0')
            await self.redis.expire(key, 60)

        return True

    async def _is_entangled(self, node: str, agent: int) -> bool:
        """Verifica se agente já está emaranhado."""
        key = await self.redis.get(f"{self.key_prefix}agent:{node}:{agent}")
        if not key:
            return False
        active = await self.redis.hget(key, 'active')
        return active == '1'
