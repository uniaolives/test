"""
AGENT MIGRATION
Implementa a lógica de transferência de agentes entre nós do cluster.
"""

import pickle
from typing import List, Any

class AgentMigrator:
    def __init__(self, node_id: str):
        self.node_id = node_id

    async def export_agents(self, agents_list: List[Any]) -> bytes:
        """Serializa agentes para transferência"""
        # Na implementação real, filtraríamos apenas os dados necessários
        # e usaríamos um formato mais eficiente que pickle (como msgpack)
        return pickle.dumps(agents_list)

    async def import_agents(self, serialized_data: bytes) -> List[Any]:
        """Desserializa agentes recebidos"""
        return pickle.loads(serialized_data)
