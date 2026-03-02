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

    async def stop(self):
        self.running = False
        if self.command_socket: self.command_socket.close()
        self.zmq_context.term()
