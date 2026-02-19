# arkhe_qutip/chain_bridge.py
import time
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class HandoverRecord:
    handover_id: str
    node_id: str
    timestamp: float
    chain_tx_hash: str
    chain_block_height: int
    metadata: Dict[str, Any]

class ArkheChainBridge:
    """
    Bridge to Arkhe(N)Chain.
    Currently implemented as a Mock for development.
    """
    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        self.ledger: List[HandoverRecord] = []
        self.node_histories: Dict[str, List[HandoverRecord]] = {}
        self.current_height = 1000

    def record_handover(self, event: Any, node_id: str) -> HandoverRecord:
        """Records a quantum handover event on the blockchain."""
        self.current_height += random.randint(1, 5)

        tx_data = f"{node_id}-{event.timestamp}-{random.random()}"
        tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()

        record = HandoverRecord(
            handover_id=str(random.randint(10000, 99999)),
            node_id=node_id,
            timestamp=event.timestamp,
            chain_tx_hash=tx_hash,
            chain_block_height=self.current_height,
            metadata=event.metadata
        )

        self.ledger.append(record)
        if node_id not in self.node_histories:
            self.node_histories[node_id] = []
        self.node_histories[node_id].append(record)

        if self.mock_mode:
            print(f"[MOCK CHAIN] Recorded handover {record.handover_id} for node {node_id} at height {record.chain_block_height}")

        return record

    def record_simulation(self, psi_initial: Any, psi_final: Any, metadata: Optional[Dict[str, Any]] = None) -> HandoverRecord:
        """Records a full quantum simulation result on the chain."""
        self.current_height += random.randint(5, 20)

        node_id = getattr(psi_final, 'node_id', 'unknown-sim')
        tx_data = f"sim-{node_id}-{time.time()}"
        tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()

        record = HandoverRecord(
            handover_id=f"SIM-{random.randint(1000, 9999)}",
            node_id=node_id,
            timestamp=time.time(),
            chain_tx_hash=tx_hash,
            chain_block_height=self.current_height,
            metadata=metadata or {}
        )

        self.ledger.append(record)
        return record

    def get_node_history(self, node_id: str) -> List[HandoverRecord]:
        """Queries the blockchain for a node's handover history."""
        return self.node_histories.get(node_id, [])
