"""
AGENT MIGRATION
Implementa a lógica de transferência de agentes entre nós do cluster.
Utiliza msgpack para serialização segura (evita vulnerabilidades do pickle).
"""

import msgpack
from typing import List, Any

class AgentMigrator:
    def __init__(self, node_id: str):
        self.node_id = node_id

    async def export_agents(self, agents_list: List[Any]) -> bytes:
        """Serializa agentes para transferência usando msgpack"""
        # Na implementação real, os agentes seriam convertidos em dicionários simples primeiro
        return msgpack.packb(agents_list)

    async def import_agents(self, serialized_data: bytes) -> List[Any]:
        """Desserializa agentes recebidos de forma segura"""
        return msgpack.unpackb(serialized_data)
