#!/usr/bin/env python3
"""
PARALLAX NODE CLIENT
Interface de nó Arkhe(n) com o controller Parallax
Suporte a Operações Quânticas Distribuídas via NCCL
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
        self.nccl_rank = -1
        self.world_size = 1
        self.nccl_id_b64 = None

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
                    'partition': self.partition
                }
                async with session.post(f"{self.controller_url}/nodes/register", json=node_info) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.nccl_rank = data.get('nccl_rank', -1)
                        self.nccl_id_b64 = data.get('nccl_unique_id')
                        self.world_size = data.get('world_size', 1)
                        logger.info(f"   ✓ Registrado (Rank: {self.nccl_rank})")
                        await self._init_quantum_distributed()
        except Exception as e:
            logger.error(f"Connect error: {e}")

        self.running = True
        asyncio.create_task(self.command_listener())
        asyncio.create_task(self.metrics_reporter())

    async def _init_quantum_distributed(self):
        lib_path = "/opt/arkhe/lib/libqhttp_dist.so"
        if os.path.exists(lib_path) and self.nccl_id_b64:
            try:
                self.qhttp_dist_lib = ctypes.CDLL(lib_path)
                # Decode NCCL ID
                nccl_id_bytes = base64.b64decode(self.nccl_id_b64)

                # Setup argtypes
                self.qhttp_dist_lib.qhttp_dist_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
                self.qhttp_dist_lib.qhttp_dist_init.restype = ctypes.c_int

                num_agents = int(os.getenv('ARKHE_AGENTS', '150'))
                ret = self.qhttp_dist_lib.qhttp_dist_init(num_agents, self.nccl_rank, self.world_size, nccl_id_bytes)
                if ret == 0:
                    logger.info("⚛️  NCCL Distributed Quantum Layer Initialized")
            except Exception as e:
                logger.warning(f"⚠️  Failed to init quantum lib: {e}")

    async def command_listener(self):
        while self.running:
            try:
                if await self.command_socket.poll(timeout=100):
                    msg = await self.command_socket.recv_json()
                    await self.handle_command(msg)
            except: pass

    async def handle_command(self, msg: dict):
        cmd = msg.get('command')
        if cmd == 'REMOTE_ENTANGLE':
            if self.node_id == msg['node_a'] or self.node_id == msg['node_b']:
                logger.info(f"⚛️  Handshaking entanglement: {msg['agent_a']} <-> {msg['agent_b']}")
                if self.qhttp_dist_lib:
                    # Logic to trigger CUDA kernels for entanglement
                    pass
        elif cmd == 'SHUTDOWN':
            self.running = False

    async def entangle_remote(self, local_agent: int, remote_node: str, remote_agent: int, bell_type: int = 0) -> bool:
        if not self.qhttp_dist_lib: return False
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    'node_a': self.node_id, 'agent_a': local_agent,
                    'node_b': remote_node, 'agent_b': remote_agent,
                    'bell_type': bell_type
                }
                async with session.post(f"{self.controller_url}/entangle/create", json=data) as resp:
                    if resp.status == 200:
                        # In real case, we'd also call the local CUDA lib here
                        # and get the remote rank from controller
                        return True
        except: return False
        return False

    async def collapse_remote(self, agent_id: int) -> Optional[int]:
        if not self.qhttp_dist_lib: return None
        measured = ctypes.c_int()
        self.qhttp_dist_lib.qhttp_collapse_remote.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        ret = self.qhttp_dist_lib.qhttp_collapse_remote(agent_id, ctypes.byref(measured))
        if ret == 0:
            # Inform controller to break entanglement record
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.controller_url}/entangle/break", json={'node': self.node_id, 'agent': agent_id})
            return measured.value
        return None

    async def metrics_reporter(self):
        while self.running:
            await asyncio.sleep(2)
            if self.simulation:
                stats = self.simulation.get_stats()
                try: await self.metrics_socket.send_json({'node_id': self.node_id, 'agents': stats['agents'], 'health': stats['avg_health']})
                except: pass

    async def disconnect(self):
        self.running = False
        if self.qhttp_dist_lib:
            try: self.qhttp_dist_lib.qhttp_dist_finalize()
            except: pass
        if self.command_socket: self.command_socket.close()
        if self.metrics_socket: self.metrics_socket.close()
        self.zmq_context.term()
