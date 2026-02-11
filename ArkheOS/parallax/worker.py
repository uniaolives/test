#!/usr/bin/env python3
"""
PARALLAX WORKER
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
from parallax.qnet_interface import QNet, QNetError

logger = logging.getLogger("Parallax.Worker")

class ParallaxWorker:
    def __init__(self,
                 node_id: str,
                 controller_url: str,
                 redis_url: str):
        self.node_id = node_id
        self.controller_url = controller_url
        self.redis_url = redis_url

        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.metrics_socket = None

        self.running = False
        self.num_agents = int(os.getenv('ARKHE_AGENTS', '150'))

        # Parallax state
        self.nccl_rank = -1
        self.world_size = 1

        # Inicializa QNet se habilitado
        self.use_qnet = os.getenv("ENABLE_QNET", "false").lower() == "true"
        self.qnet: Optional[QNet] = None

        if self.use_qnet:
            try:
                self.qnet = QNet()
                logger.info(f"✅ QNet enabled for {node_id}")
            except QNetError as e:
                logger.warning(f"⚠️ QNet init failed, falling back to ZeroMQ: {e}")
                self.use_qnet = False

    async def start(self):
        logger.info(f"🔌 Iniciando Parallax Worker {self.node_id}")
        self.command_socket = self.zmq_context.socket(zmq.SUB)
        self.command_socket.connect("tcp://parallax-controller:5555")
        self.command_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.metrics_socket = self.zmq_context.socket(zmq.PUSH)
        self.metrics_socket.connect("tcp://parallax-controller:5556")

        try:
            async with aiohttp.ClientSession() as session:
                node_info = {
                    'node_id': self.node_id, 'hostname': os.uname().nodename,
                }
                async with session.post(f"{self.controller_url}/nodes/register", json=node_info) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.nccl_rank = data.get('nccl_rank', -1)
                        self.world_size = data.get('world_size', 1)
        except Exception as e:
            logger.error(f"Connect error: {e}")

        self.running = True
        asyncio.create_task(self.command_listener())

    async def command_listener(self):
        while self.running:
            try:
                if await self.command_socket.poll(timeout=100):
                    msg = await self.command_socket.recv_json()
                    if msg.get('command') == 'SHUTDOWN': self.running = False
            except: pass

    async def send_critical_message(self, target_node: str, msg_type: str, data: dict):
        """
        Envia mensagem crítica via DPDK se disponível, senão ZeroMQ.

        Mensagens críticas:
        - quantum_collapse
        - entanglement_request
        - qec_syndrome
        """
        import msgpack

        # Serializa
        packet = msgpack.packb({
            "from": self.node_id,
            "to": target_node,
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        })

        # Rota via QNet se disponível
        if self.use_qnet and self.qnet:
            try:
                self.qnet.send(packet)
                logger.debug(f"📡 Sent {msg_type} via QNet to {target_node}")
                return
            except QNetError as e:
                logger.warning(f"QNet send failed, falling back: {e}")

        # Fallback para ZeroMQ
        await self._zmq_send(target_node, packet)

    async def _zmq_send(self, target_node: str, packet: bytes):
        # Implementation of ZeroMQ fallback send
        # In a real scenario, this would use a DEALER socket or similar
        logger.debug(f"ZMQ fallback: sending to {target_node}")
        pass

    async def stop(self):
        self.running = False
        if self.command_socket: self.command_socket.close()
        if self.qnet: self.qnet.close()
        self.zmq_context.term()
