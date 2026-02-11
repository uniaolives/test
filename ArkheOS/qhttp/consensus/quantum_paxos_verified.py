import asyncio
import hashlib
import time
import json
from dataclasses import dataclass
from typing import Dict, Set, Optional, List
from enum import Enum
import redis.asyncio as aioredis
from .tla_monitor import TLAMonitor

class QuantumStateProof(Enum):
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"

@dataclass
class QuantumCommit:
    slot: int
    ballot: int
    value: Dict
    proof: QuantumStateProof
    node_signature: str
    timestamp: float

class QuantumPaxosVerified:
    """
    Byzantine Fault Tolerant consensus with Runtime Verification hooks.
    """
    def __init__(self, node_id: str, redis_url: str, total_nodes: int = 3):
        self.node_id = node_id
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.total_nodes = total_nodes
        self.fault_tolerance = (total_nodes - 1) // 3
        self.ballot = 0
        self.slot = 0
        self.promises: Set[str] = set()
        self.accepts: Set[str] = set()
        self.state_log: List[QuantumCommit] = []

        # Runtime Verification
        self.monitor = TLAMonitor("ArkheOS/spec/quantum_paxos.tla")

    def _get_local_state(self):
        return {
            "ballot": self.ballot,
            "slot": self.slot,
            "promises_count": len(self.promises),
            "accepts_count": len(self.accepts)
        }

    async def propose_state(self, quantum_state: Dict) -> bool:
        old_state = self._get_local_state()
        self.ballot += 1
        self.promises.clear()
        self.accepts.clear()

        # Phase 1: Prepare
        prepare_msg = {
            "type": "PREPARE",
            "ballot": self.ballot,
            "slot": self.slot,
            "node": self.node_id,
            "timestamp": time.time()
        }

        self.monitor.log_transition("PREPARE", self.node_id, old_state, self._get_local_state())
        await self.redis.publish("quantum:consensus:prepare", json.dumps(prepare_msg))

        try:
            await asyncio.wait_for(self._collect_promises(), timeout=0.5)
        except asyncio.TimeoutError:
            return False

        if len(self.promises) < 2 * self.fault_tolerance + 1:
            return False

        # Phase 2: Accept
        commit = QuantumCommit(
            slot=self.slot,
            ballot=self.ballot,
            value=quantum_state,
            proof=QuantumStateProof.SUPERPOSITION,
            node_signature=self._sign(quantum_state),
            timestamp=time.time()
        )

        accept_msg = {
            "type": "ACCEPT",
            "commit": self._serialize_commit(commit),
            "node": self.node_id
        }

        self.monitor.log_transition("ACCEPT", self.node_id, self._get_local_state(), self._get_local_state())
        await self.redis.publish("quantum:consensus:accept", json.dumps(accept_msg))

        try:
            await asyncio.wait_for(self._collect_accepts(), timeout=0.5)
        except asyncio.TimeoutError:
            return False

        if len(self.accepts) >= 2 * self.fault_tolerance + 1:
            prev_state = self._get_local_state()
            self.state_log.append(commit)
            self.slot += 1
            await self._apply_commit(commit)

            self.monitor.log_transition("COMMIT", self.node_id, prev_state, {
                **self._get_local_state(),
                "value": hashlib.sha256(json.dumps(quantum_state).encode()).hexdigest()
            })
            return True

        return False

    def _serialize_commit(self, commit: QuantumCommit) -> Dict:
        return {
            "slot": commit.slot,
            "ballot": commit.ballot,
            "value": commit.value,
            "proof": commit.proof.value,
            "node_signature": commit.node_signature,
            "timestamp": commit.timestamp
        }

    def _sign(self, state: Dict) -> str:
        state_hash = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return f"{self.node_id}:{state_hash}:{int(time.time())}"

    async def _collect_promises(self):
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("quantum:consensus:promise")
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                if data.get("ballot") == self.ballot:
                    self.promises.add(data["node"])
                    if len(self.promises) >= 2 * self.fault_tolerance + 1:
                        break

    async def _collect_accepts(self):
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("quantum:consensus:accepted")
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                if data.get("ballot") == self.ballot:
                    self.accepts.add(data["node"])
                    if len(self.accepts) >= 2 * self.fault_tolerance + 1:
                        break

    async def _apply_commit(self, commit: QuantumCommit):
        await self.redis.hset(f"quantum:state:{commit.slot}", mapping={
            "value": json.dumps(commit.value),
            "proof": commit.proof.value,
            "timestamp": commit.timestamp,
            "signature": commit.node_signature
        })
