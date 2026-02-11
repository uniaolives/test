#!/usr/bin/env python3
"""
PARALLAX CONTROLLER v2.0
Orquestrador global para cluster Arkhe(n) distribuído
Implementa scheduling Hebbiano e Entanglement Registry
"""

import asyncio
import json
import logging
import time
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from .entanglement_registry import EntanglementRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Parallax.Controller")

@dataclass
class ArkheNode:
    """Representa um nó Arkhe(n) no cluster"""
    node_id: str
    hostname: str
    address: str
    port: int
    partition: Tuple[int, int, int]
    resources: Dict[str, float] = field(default_factory=dict)
    agents_count: int = 0
    health_score: float = 1.0
    last_heartbeat: float = 0.0
    is_active: bool = False
    avg_latency_ms: float = 0.0
    throughput: float = 0.0
    nccl_rank: int = -1

class ParallaxController:
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.nodes: Dict[str, ArkheNode] = {}
        self.redis: Optional[Any] = None
        self.redis_url = redis_url
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.metrics_socket = None
        self.running = False
        self.registry = None
        self.nccl_unique_id_b64 = None

    async def initialize(self):
        logger.info("🎛️  Inicializando Parallax Controller v2.0")
        if redis:
            try:
                self.redis = await redis.from_url(self.redis_url, decode_responses=True)
                await self.redis.ping()
                self.registry = EntanglementRegistry(self.redis)
                logger.info("   ✓ Redis e Entanglement Registry ativos")
            except Exception as e:
                logger.error(f"   ✗ Falha no Redis: {e}")

        # Simula geração de NCCL Unique ID
        # Em produção: import nccl; self.nccl_id = nccl.get_unique_id()
        self.nccl_unique_id_b64 = base64.b64encode(b"MOCKED_NCCL_UNIQUE_ID_128BIT_LENGTH_!!!").decode()

        self.command_socket = self.zmq_context.socket(zmq.PUB)
        self.command_socket.bind("tcp://*:5555")
        self.metrics_socket = self.zmq_context.socket(zmq.PULL)
        self.metrics_socket.bind("tcp://*:5556")

        asyncio.create_task(self.heartbeat_monitor())
        asyncio.create_task(self.metrics_collector())
        self.running = True
        logger.info("✅ Controller operacional")

    async def register_node(self, node_data: Dict) -> Dict:
        node_id = node_data['node_id']
        if node_id in self.nodes:
            node = self.nodes[node_id]
        else:
            # Atribui rank NCCL sequencial
            rank = len(self.nodes)
            node = ArkheNode(
                node_id=node_id,
                hostname=node_data.get('hostname', ''),
                address=node_data.get('address', ''),
                port=node_data.get('port', 8000),
                partition=tuple(node_data.get('partition', (0,0,0))),
                nccl_rank=rank
            )
            self.nodes[node_id] = node
            if self.redis:
                await self.redis.set(f"parallax:node:{node_id}:rank", rank)

        node.last_heartbeat = time.time()
        node.is_active = True

        logger.info(f"🖥️  Nó registrado: {node_id} (Rank: {node.nccl_rank})")
        return {
            "success": True,
            "nccl_rank": node.nccl_rank,
            "nccl_unique_id": self.nccl_unique_id_b64,
            "world_size": len(self.nodes) # Simplificado
        }

    async def heartbeat_monitor(self):
        while self.running:
            current_time = time.time()
            for node_id, node in list(self.nodes.items()):
                if node.is_active and current_time - node.last_heartbeat > 30:
                    logger.warning(f"💔 Heartbeat perdido: {node_id}")
                    node.is_active = False
            await asyncio.sleep(5)

    async def metrics_collector(self):
        while self.running:
            try:
                if await self.metrics_socket.poll(timeout=1000):
                    msg = await self.metrics_socket.recv_json()
                    node_id = msg.get('node_id')
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        node.agents_count = msg.get('agents', 0)
                        node.last_heartbeat = time.time()
                        node.is_active = True
            except: pass

    async def shutdown(self):
        self.running = False
        if self.command_socket: self.command_socket.close()
        if self.metrics_socket: self.metrics_socket.close()
        self.zmq_context.term()
        if self.redis: await self.redis.close()

controller = ParallaxController()
app = FastAPI(title="Parallax Controller")

@app.on_event("startup")
async def startup(): await controller.initialize()

@app.on_event("shutdown")
async def shutdown_event(): await controller.shutdown()

@app.get("/health")
async def health():
    active_nodes = [n for n in controller.nodes.values() if n.is_active]
    return {
        'status': 'online',
        'nodes_active': len(active_nodes),
        'total_agents': sum(n.agents_count for n in active_nodes)
    }

@app.post("/nodes/register")
async def register_node(node_data: Dict):
    return await controller.register_node(node_data)

@app.get("/nodes/{node_id}/rank")
async def get_node_rank(node_id: str):
    if node_id not in controller.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node": node_id, "rank": controller.nodes[node_id].nccl_rank}

@app.get("/nccl/get_unique_id")
async def get_nccl_unique_id():
    return {"unique_id": controller.nccl_unique_id_b64}

@app.post("/entangle/create")
async def create_entanglement(payload: Dict):
    if not controller.registry:
        raise HTTPException(status_code=503, detail="Registry not active")

    success = await controller.registry.create_pair(
        payload['node_a'], int(payload['agent_a']),
        payload['node_b'], int(payload['agent_b']),
        payload.get('bell_type', 0)
    )

    if success and controller.command_socket:
        await controller.command_socket.send_json({
            'command': 'REMOTE_ENTANGLE',
            **payload
        })

    return {"success": success}

@app.post("/entangle/break")
async def break_entanglement(payload: Dict):
    if not controller.registry:
        raise HTTPException(status_code=503)
    success = await controller.registry.break_pair(payload['node'], int(payload['agent']))
    return {"success": success}

@app.get("/entangle/partner")
async def get_partner(node: str, agent: int):
    if not controller.registry:
        raise HTTPException(status_code=503)
    partner = await controller.registry.get_partner(node, agent)
    if partner:
        return {"node": partner[0], "agent": partner[1], "bell_type": partner[2]}
    raise HTTPException(status_code=404, detail="No entanglement found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
