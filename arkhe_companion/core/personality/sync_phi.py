# core/python/arkhe/companion/sync_engine.py
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

@dataclass
class HybridClock:
    """
    Combina relógio físico (para ordenação total quando possível)
    com contador lógico (para causalidade quando clocks divergem).
    """
    node_id: str
    physical: int = 0  # timestamp físico (ms desde epoch)
    logical: int = 0   # contador para eventos no mesmo ms

    def tick(self) -> Tuple[int, int, str]:
        now = int(time.time() * 1000)
        if now == self.physical:
            self.logical += 1
        else:
            self.physical = now
            self.logical = 0
        return (self.physical, self.logical, self.node_id)

    def compare(self, other_phys: int, other_log: int, other_id: str) -> int:
        # Ordenação: físico primeiro, depois lógico, depois ID (desempate)
        if self.physical != other_phys:
            return self.physical - other_phys
        if self.logical != other_log:
            return self.logical - other_log
        if self.node_id < other_id:
            return -1
        if self.node_id > other_id:
            return 1
        return 0

@dataclass
class CRDTOp:
    type: str
    value: Any
    timestamp: Tuple[int, int, str]

@dataclass
class CRDTDelta:
    device_id: str
    store_name: str
    operations: List[CRDTOp]
    timestamp: Tuple[int, int, str]

class LWWRegister:
    """Last-Write-Wins Register para valor escalar."""
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.ts_phys = 0
        self.ts_log = 0
        self.ts_id = ""

    def set(self, new_value: Any, ts: Tuple[int, int, str]) -> bool:
        phys, log, node_id = ts
        # Comparação Last-Write-Wins
        is_newer = False
        if phys > self.ts_phys:
            is_newer = True
        elif phys == self.ts_phys:
            if log > self.ts_log:
                is_newer = True
            elif log == self.ts_log:
                if node_id > self.ts_id:
                    is_newer = True

        if is_newer:
            self.value = new_value
            self.ts_phys = phys
            self.ts_log = log
            self.ts_id = node_id
            return True
        return False

    def merge_delta(self, delta: CRDTDelta) -> bool:
        changed = False
        for op in delta.operations:
            if op.type == 'set':
                if self.set(op.value, op.timestamp):
                    changed = True
        return changed

class DeviceSyncEngine:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.clock = HybridClock(device_id)
        self.stores: Dict[str, Any] = {
            'personality_phi': LWWRegister(initial_value=0.618),
            'short_memory': LWWRegister(),
        }
        self.pending_deltas = deque(maxlen=1000)
        self.online_peers: List[Any] = [] # Mock for peers
        self.subscriptions: Dict[str, List[Any]] = {}

    def subscribe(self, store_name: str, callback: Any):
        if store_name not in self.subscriptions:
            self.subscriptions[store_name] = []
        self.subscriptions[store_name].append(callback)

    def broadcast_delta(self, store_name: str, delta: CRDTDelta):
        # Em produção, enviaria para peers via rede
        # Aqui simulamos o armazenamento local e notificação de inscritos
        self.pending_deltas.append(delta)
        if store_name in self.subscriptions:
            for cb in self.subscriptions[store_name]:
                cb(delta)

    def local_update(self, store_name: str, value: Any) -> Optional[CRDTDelta]:
        if store_name not in self.stores:
            return None

        store = self.stores[store_name]
        ts = self.clock.tick()

        if hasattr(store, 'set'):
            if store.set(value, ts):
                delta = CRDTDelta(
                    device_id=self.device_id,
                    store_name=store_name,
                    operations=[CRDTOp(type='set', value=value, timestamp=ts)],
                    timestamp=ts
                )
                self.broadcast_delta(store_name, delta)
                return delta
        return None

    def receive_delta(self, delta: CRDTDelta):
        if delta.store_name in self.stores:
            store = self.stores[delta.store_name]
            if store.merge_delta(delta):
                # Se mudou, notifica inscritos locais
                if delta.store_name in self.subscriptions:
                    for cb in self.subscriptions[delta.store_name]:
                        cb(delta)

class CompactionEngine:
    """
    Sumariza operações antigas em snapshots criptograficamente selados.
    """
    def __init__(self, secret_key: str = "arkhe_secret"):
        self.secret_key = secret_key

    def _hash_state(self, data: Any) -> str:
        return hashlib.sha256(str(data).encode() + self.secret_key.encode()).hexdigest()

    def compact(self, history: List[CRDTDelta], retention_seconds: int = 3600) -> Tuple[Dict, List[CRDTDelta]]:
        cutoff = time.time() - retention_seconds

        old_deltas = [d for d in history if d.timestamp[0] / 1000.0 < cutoff]
        recent_deltas = [d for d in history if d.timestamp[0] / 1000.0 >= cutoff]

        if not old_deltas:
            return {}, history

        # Simplificação: O snapshot contém o hash do estado acumulado
        snapshot = {
            'state_hash': self._hash_state(old_deltas),
            'compacted_count': len(old_deltas),
            'timestamp': time.time(),
            'last_included_ts': old_deltas[-1].timestamp
        }

        return snapshot, recent_deltas
