#!/usr/bin/env python3
"""
PARALLAX CONTROLLER v2.0
Orquestrador global para cluster Arkhe(n) distribuído
Implementa scheduling Hebbiano e balanceamento de carga biológico
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
try:
    import redis.asyncio as redis
except ImportError:
    # Fallback for systems where redis-py is older or not installed
    redis = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Parallax.Controller")

@dataclass
class ArkheNode:
    """Representa um nó Arkhe(n) no cluster"""
    node_id: str
    hostname: str
    address: str
    port: int
    partition: Tuple[int, int, int]  # Octante 3D atribuído
    resources: Dict[str, float] = field(default_factory=dict)
    agents_count: int = 0
    health_score: float = 1.0
    last_heartbeat: float = 0.0
    is_active: bool = False

    # Métricas de performance
    avg_latency_ms: float = 0.0
    throughput: float = 0.0  # agentes/tick

class ParallaxController:
    """
    Controller central estilo Parallax para orquestração de nós Arkhe(n)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.nodes: Dict[str, ArkheNode] = {}
        self.redis: Optional[Any] = None
        self.redis_url = redis_url

        # Contexto ZeroMQ para comunicação de alta performance
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None  # PUB para comandos globais
        self.metrics_socket = None  # PULL para métricas dos nós

        # Estado do campo global (visualização apenas)
        self.global_field_stats = {
            'total_agents': 0,
            'total_bonds': 0,
            'avg_health': 0.0,
            'field_entropy': 0.0
        }

        # Configurações de particionamento
        self.halo_size = 5  # Células de sobreposição entre nós
        self.space_size = (100, 100, 100)  # Tamanho total do campo

        self.running = False

    async def initialize(self):
        """Inicializa o controller"""
        logger.info("🎛️  Inicializando Parallax Controller v2.0")

        # Conecta ao Redis
        if redis:
            try:
                self.redis = await redis.from_url(self.redis_url, decode_responses=True)
                await self.redis.ping()
                logger.info("   ✓ Redis conectado")
            except Exception as e:
                logger.error(f"   ✗ Falha no Redis: {e}")
                # We can continue without redis for non-persistence features

        # Inicializa sockets ZeroMQ
        self.command_socket = self.zmq_context.socket(zmq.PUB)
        self.command_socket.bind("tcp://*:5555")

        self.metrics_socket = self.zmq_context.socket(zmq.PULL)
        self.metrics_socket.bind("tcp://*:5556")

        logger.info("   ✓ ZeroMQ ativo (PUB:5555, PULL:5556)")

        # Agenda tarefas de background
        asyncio.create_task(self.heartbeat_monitor())
        asyncio.create_task(self.metrics_collector())
        asyncio.create_task(self.load_balancer())

        self.running = True
        logger.info("✅ Controller operacional")

    async def register_node(self, node: ArkheNode) -> bool:
        """Registra um novo nó no cluster"""
        # Verifica se há conflito de partição
        for existing in self.nodes.values():
            if existing.partition == node.partition and existing.is_active:
                logger.warning(f"Conflito de partição: {node.node_id} vs {existing.node_id}")
                return False

        self.nodes[node.node_id] = node

        # Publica no Redis para descoberta
        if self.redis:
            await self.redis.hset(f"parallax:node:{node.node_id}", mapping={
                'hostname': node.hostname,
                'address': node.address,
                'partition': json.dumps(node.partition),
                'agents': node.agents_count,
                'health': node.health_score
            })

        logger.info(f"🖥️  Nó registrado: {node.node_id} @ {node.address} "
                   f"(partição {node.partition})")
        return True

    async def unregister_node(self, node_id: str):
        """Remove nó do cluster"""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            if self.redis:
                await self.redis.delete(f"parallax:node:{node_id}")
            logger.info(f"🖥️  Nó removido: {node_id}")

    async def heartbeat_monitor(self):
        """Monitora saúde dos nós via heartbeat"""
        while self.running:
            current_time = time.time()
            dead_nodes = []

            for node_id, node in self.nodes.items():
                if current_time - node.last_heartbeat > 30:  # 30s timeout
                    logger.warning(f"💔 Heartbeat perdido: {node_id}")
                    node.is_active = False
                    dead_nodes.append(node_id)

            # Tenta recuperar ou realoca nós mortos
            for node_id in dead_nodes:
                await self.handle_node_failure(node_id)

            await asyncio.sleep(5)

    async def metrics_collector(self):
        """Coleta métricas dos nós via ZeroMQ"""
        while self.running:
            try:
                # Non-blocking receive com timeout
                if await self.metrics_socket.poll(timeout=1000):
                    msg = await self.metrics_socket.recv_json()
                    node_id = msg.get('node_id')

                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        node.agents_count = msg.get('agents', 0)
                        node.health_score = msg.get('health', 1.0)
                        node.last_heartbeat = time.time()
                        node.is_active = True
                        node.avg_latency_ms = msg.get('latency_ms', 0)

                        # Atualiza Redis
                        if self.redis:
                            await self.redis.hset(f"parallax:node:{node_id}", mapping={
                                'agents': node.agents_count,
                                'health': node.health_score,
                                'latency': node.avg_latency_ms
                            })

            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")

    async def load_balancer(self):
        """
        Balanceamento de carga Hebbiano:
        - Nós com alta sinergia (bonds) ficam próximos
        - Migra agentes de nós sobrecarregados
        """
        while self.running:
            await asyncio.sleep(10)  # A cada 10 segundos

            if len(self.nodes) < 2:
                continue

            # Calcula carga média
            active_nodes = [n for n in self.nodes.values() if n.is_active]
            if not active_nodes:
                continue

            avg_load = np.mean([n.agents_count for n in active_nodes])

            # Identifica nós sobrecarregados e subutilizados
            overloaded = [n for n in active_nodes if n.agents_count > avg_load * 1.3]
            underloaded = [n for n in active_nodes if n.agents_count < avg_load * 0.7]

            # Orquestra migrações
            for src in overloaded:
                if underloaded:
                    dst = underloaded.pop(0)
                    await self.orchestrate_migration(src, dst)

    async def orchestrate_migration(self, src: ArkheNode, dst: ArkheNode):
        """Orquestra migração de agentes entre nós"""
        # Calcula quantos agentes migrar (Hebbiano: mantém bonds)
        migrants = int((src.agents_count - dst.agents_count) * 0.2)

        logger.info(f"🔄 Migração: {migrants} agentes de {src.node_id} → {dst.node_id}")

        # Comando via ZeroMQ
        await self.command_socket.send_json({
            'command': 'MIGRATE_OUT',
            'target_node': dst.node_id,
            'count': migrants,
            'priority': 'hebbian'  # Mantém conexões sociais
        })

    async def handle_node_failure(self, node_id: str):
        """Lida com falha de nó - realoca agentes"""
        logger.error(f"🔥 Falha detectada no nó: {node_id}")

        node = self.nodes.get(node_id)
        if not node:
            return

        # Encontra nós vizinhos (partições adjacentes)
        neighbors = self.find_neighbor_partitions(node.partition)

        # Redistribui carga
        for neighbor in neighbors:
            if neighbor.is_active:
                await self.command_socket.send_json({
                    'command': 'ADOPT_PARTITION',
                    'failed_node': node_id,
                    'partition': node.partition
                })
                break

        await self.unregister_node(node_id)

    def find_neighbor_partitions(self, partition: Tuple[int, int, int]) -> List[ArkheNode]:
        """Encontra nós com partições espacialmente adjacentes"""
        # Implementação simplificada: retorna todos os nós ativos
        return [n for n in self.nodes.values() if n.is_active]

    async def get_global_state(self) -> Dict:
        """Retorna estado consolidado de todo o cluster"""
        active_nodes = [n for n in self.nodes.values() if n.is_active]

        return {
            'cluster': {
                'nodes_total': len(self.nodes),
                'nodes_active': len(active_nodes),
                'partitions': [n.partition for n in active_nodes]
            },
            'agents': {
                'total': sum(n.agents_count for n in active_nodes),
                'capacity': sum(n.resources.get('max_agents', 1000) for n in active_nodes),
                'avg_health': np.mean([n.health_score for n in active_nodes]) if active_nodes else 0
            },
            'performance': {
                'avg_latency_ms': np.mean([n.avg_latency_ms for n in active_nodes]) if active_nodes else 0,
                'total_throughput': sum(n.throughput for n in active_nodes)
            }
        }

    async def shutdown(self):
        """Desliga o controller graciosamente"""
        logger.info("🛑 Desligando Parallax Controller...")
        self.running = False

        # Notifica todos os nós
        if self.command_socket:
            try:
                await self.command_socket.send_json({
                    'command': 'SHUTDOWN',
                    'reason': 'controller_stop'
                })
            except: pass

        await asyncio.sleep(1)
        if self.command_socket: self.command_socket.close()
        if self.metrics_socket: self.metrics_socket.close()
        self.zmq_context.term()

        if self.redis:
            await self.redis.close()

# FastAPI App para o Controller
controller = ParallaxController()
app = FastAPI(title="Parallax Controller", version="2.0.0")

@app.on_event("startup")
async def startup():
    await controller.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await controller.shutdown()

@app.get("/health")
async def health():
    return await controller.get_global_state()

@app.post("/nodes/register")
async def register_node(node: ArkheNode):
    success = await controller.register_node(node)
    return {"success": success}

@app.get("/nodes")
async def list_nodes():
    return {
        node_id: {
            'partition': node.partition,
            'agents': node.agents_count,
            'health': node.health_score,
            'active': node.is_active
        }
        for node_id, node in controller.nodes.items()
    }

@app.post("/command/broadcast")
async def broadcast_command(cmd: dict):
    if controller.command_socket:
        await controller.command_socket.send_json(cmd)
        return {"sent": True}
    return {"sent": False, "error": "Command socket not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
