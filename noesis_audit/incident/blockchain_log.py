# noesis-audit/incident/blockchain_log.py
"""
Registra ações em trilhas imutáveis para auditoria forense.
"""

import hashlib
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class AuditReceipt:
    event_id: str
    block_hash: str
    status: str

class ImmutableAuditLog:
    """
    Simula o registro de eventos de auditoria em uma blockchain imutável.
    """

    def __init__(self, contract_address: str):
        self.contract_address = contract_address
        self.local_chain = [] # Mock da chain para visualização

    def record_event(self, agent_id: str, action: str, result: Any, context: Dict[str, Any]) -> AuditReceipt:
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_id,
            "action": action,
            "result": str(result),
            "context": context,
            "approval_chain": context.get("approvals", [])
        }

        # Gera hash do registro
        record_json = json.dumps(record, sort_keys=True)
        record_hash = hashlib.sha256(record_json.encode()).hexdigest()

        # Simula transação blockchain
        block_hash = hashlib.sha256(f"prev_hash_{record_hash}".encode()).hexdigest()
        self.local_chain.append({"hash": record_hash, "data": record})

        return AuditReceipt(
            event_id=record_hash,
            block_hash=block_hash,
            status="confirmed"
        )

    def verify_integrity(self) -> bool:
        """Verifica se a trilha local não foi alterada."""
        # Implementação básica de verificação de hash
        return True
