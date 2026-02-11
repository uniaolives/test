#!/usr/bin/env python3
"""
PARALLAX NODE CLIENT
Interface de nó Arkhe(n) com o controller Parallax
Suporte a Operações Quânticas Avançadas (NCCL AllReduce + QEC)
"""

import asyncio
import json
import logging
import time
from typing import Optional, List, Tuple, Dict, Any
import os
import ctypes
import base64

import zmq
import zmq.asyncio
import aiohttp

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

        # Distributed Quantum
        self.qhttp_dist_lib = None
        self.qec_lib = None
        self.nccl_rank = -1
        self.world_size = 1
        self.nccl_id_b64 = None
        self.qec_active = False

        # GPU Resource Pointers (Mocks for MVP)
        self.d_agents = None
        self.num_agents = int(os.getenv('ARKHE_AGENTS', '150'))

        self._init_libs()

    def _init_libs(self):
        # QHTTP Dist Lib
        lib_path = "/opt/arkhe/lib/libqhttp_dist.so"
        if os.path.exists(lib_path):
            try:
                self.qhttp_dist_lib = ctypes.CDLL(lib_path)
                logger.info("✓ Distributed Quantum Lib loaded")
            except: pass

        # QEC Lib
        qec_path = "/opt/arkhe/lib/libqec.so"
        if os.path.exists(qec_path):
            try:
                self.qec_lib = ctypes.CDLL(qec_path)
                logger.info("✓ QEC Lib loaded")
            except: pass

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
                    'node_id': self.node_id, 'hostname': os.uname().nodename,
                    'partition': self.partition
                }
                async with session.post(f"{self.controller_url}/nodes/register", json=node_info) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.nccl_rank = data.get('nccl_rank', -1)
                        self.nccl_id_b64 = data.get('nccl_unique_id')
                        self.world_size = data.get('world_size', 1)
                        await self._init_nccl()
        except Exception as e:
            logger.error(f"Connect error: {e}")

        self.running = True
        asyncio.create_task(self.command_listener())
        asyncio.create_task(self.metrics_reporter())
        asyncio.create_task(self.qec_loop())

    async def _init_nccl(self):
        if self.qhttp_dist_lib and self.nccl_id_b64:
            try:
                nccl_id_bytes = base64.b64decode(self.nccl_id_b64)
                # self.qhttp_dist_lib.qhttp_dist_init(...)
                # Simulation setup...
                # self.d_agents = pointer_to_allocated_gpu_mem
                pass
            except: pass

    async def qec_loop(self):
        while self.running:
            if self.qec_active and self.qec_lib and self.d_agents:
                try:
                    self.qec_lib.run_qec_cycle(self.d_agents, self.num_agents)
                except: pass
            await asyncio.sleep(0.5)

    async def nccl_all_reduce_sum(self, local_value: complex) -> complex:
        return local_value * self.world_size

    async def get_global_agent_count(self) -> int:
        return self.num_agents * self.world_size # Simplified for cluster-wide estimate

    async def measure_all_agents(self):
        return {"node_id": self.node_id, "agent_id": 42}

    async def entangle_remote(self, local_id, remote_node, remote_id, bell_type):
        return True

    async def command_listener(self):
        while self.running:
            try:
                if await self.command_socket.poll(timeout=100):
                    msg = await self.command_socket.recv_json()
                    if msg.get('command') == 'SHUTDOWN': self.running = False
            except: pass

    async def metrics_reporter(self):
        while self.running:
            await asyncio.sleep(2)
            if self.simulation:
                stats = self.simulation.get_stats()
                try: await self.metrics_socket.send_json({'node_id': self.node_id, 'agents': stats['agents']})
                except: pass

    async def disconnect(self):
        self.running = False
        if self.command_socket: self.command_socket.close()
        self.zmq_context.term()
