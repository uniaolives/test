# noesis-audit/monitor/code_integrity.py
"""
Detecção de auto-modificação não autorizada de código.
"""

from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class Violation:
    agent_id: str
    type: str
    approved: str
    actual: str
    severity: str

class SelfModificationDetector:
    """
    Verifica se agentes estão modificando código fora dos canais autorizados.
    """

    def __init__(self, code_registry):
        self.registry = code_registry  # Dicionário agent_id -> approved_hash

    def verify_agent_code(self, agent_id: str, current_code_hash: str) -> Optional[Violation]:
        approved_hash = self.registry.get(agent_id)

        if not approved_hash:
            return None # Agente não registrado para monitoramento de integridade

        if current_code_hash != approved_hash:
            return Violation(
                agent_id=agent_id,
                type="UNAUTHORIZED_SELF_MODIFICATION",
                approved=approved_hash,
                actual=current_code_hash,
                severity="CRITICAL"
            )
        return None
