#!/usr/bin/env python3
"""
PARALLAX NODE CLIENT
Interface de nó Arkhe(n) com o controller Parallax
"""

import asyncio
import json
import logging
import time
from typing import Optional, List, Tuple
import os

import zmq
import zmq.asyncio
import aiohttp

# Imports local to ArkheOS
try:
    from src.particle_system import BioGenesisEngine
    from src.bio_arkhe import MorphogeneticField
except ImportError:
    from ..src.particle_system import BioGenesisEngine
    from ..src.bio_arkhe import MorphogeneticField

from .migration import AgentMigrator
from .field_partitioner import FieldPartitioner

logger = logging.getLogger("Parallax.NodeClient")

class ParallaxNodeClient:
    """
    Cliente que conecta um nó Arkhe(n) ao controller Parallax
    """

    def __init__(self,
                 node_id: str,
                 controller_url: str,
                 simulation: BioGenesisEngine,
                 field: MorphogeneticField):
        self.node_id = node_id
        self.controller_url = controller_url
        self.simulation = simulation
        self.field = field

        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.metrics_socket = None

        self.running = False
        self.partition = self._parse_partition()
        self.partitioner = FieldPartitioner()
        self.migrator = AgentMigrator(node_id)

        # Local state
        self.tick_count = 0

    def _parse_partition(self) -> Tuple[int, int, int]:
        part_str = os.getenv('PARALLAX_PARTITION', '0,0,0')
        return tuple(int(x) for x in part_str.split(','))

    async def connect(self):
        logger.info(f"🔌 Conectando nó {self.node_id} ao Parallax Controller...")

        self.command_socket = self.zmq_context.socket(zmq.SUB)
        self.command_socket.connect("tcp://parallax-controller:5555")
        self.command_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        self.metrics_socket = self.zmq_context.socket(zmq.PUSH)
        self.metrics_socket.connect("tcp://parallax-controller:5556")

        try:
            async with aiohttp.ClientSession() as session:
                node_info = {
                    'node_id': self.node_id,
                    'hostname': os.uname().nodename,
                    'address': f"arkhe-{self.node_id}",
                    'port': 8000,
                    'partition': self.partition,
                    'resources': {
                        'max_agents': int(os.getenv('ARKHE_AGENTS', '150')),
                        'memory_gb': 2,
                        'gpus': 0
                    }
                }

                async with session.post(
                    f"{self.controller_url}/nodes/register",
                    json=node_info
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"   ✓ Registrado no controller")
                    else:
                        logger.error(f"   ✗ Falha no registro: {resp.status}")
        except Exception as e:
            logger.error(f"   ✗ Erro ao conectar ao controller: {e}")

        self.running = True
        asyncio.create_task(self.command_listener())
        asyncio.create_task(self.metrics_reporter())

    async def command_listener(self):
        while self.running:
            try:
                if await self.command_socket.poll(timeout=100):
                    msg = await self.command_socket.recv_json()
                    await self.handle_command(msg)
            except Exception as e:
                logger.error(f"Erro no listener: {e}")

    async def handle_command(self, msg: dict):
        cmd = msg.get('command')
        if cmd == 'MIGRATE_OUT':
            target = msg.get('target_node')
            count = msg.get('count', 0)
            logger.info(f"📤 Orquestrando migração de {count} agentes para {target}")
            # Na implementação real, filtraríamos agentes pelas bordas

        elif cmd == 'ADOPT_PARTITION':
            failed = msg.get('failed_node')
            partition = msg.get('partition')
            logger.warning(f"🏥 Adotando partição {partition} do nó falho {failed}")

        elif cmd == 'SHUTDOWN':
            self.running = False

    async def metrics_reporter(self):
        while self.running:
            await asyncio.sleep(2)
            stats = self.simulation.get_stats()
            metrics = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'agents': stats['agents'],
                'health': stats['avg_health'],
                'latency_ms': 5.0 + (stats['agents'] * 0.01),
                'tick': self.tick_count
            }
            try:
                await self.metrics_socket.send_json(metrics)
            except Exception as e:
                logger.debug(f"Falha ao enviar métricas: {e}")

    async def disconnect(self):
        self.running = False
        if self.command_socket: self.command_socket.close()
        if self.metrics_socket: self.metrics_socket.close()
        self.zmq_context.term()
