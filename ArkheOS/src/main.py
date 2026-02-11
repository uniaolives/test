#!/usr/bin/env python3
"""
ARKHE(N) BOOTLOADER v2.3-QUANTUM-STABLE
Suporte a QEC (Surface Code) e Grover Distribuído.
"""

import argparse
import asyncio
import logging
import time
import threading
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
import colorlog

# Imports Arkhe
try:
    from .particle_system import BioGenesisEngine
    from .shared_memory import SharedFieldManager
    from .mcp_server import create_mcp_server
    BIOGENESIS_LOADED = True
except ImportError:
    try:
        from particle_system import BioGenesisEngine
        from shared_memory import SharedFieldManager
        from mcp_server import create_mcp_server
        BIOGENESIS_LOADED = True
    except:
        BIOGENESIS_LOADED = False

# Import Parallax
try:
    from parallax.node_client import ParallaxNodeClient
    PARALLAX_LOADED = True
except ImportError:
    PARALLAX_LOADED = False

# Import Quantum Adv
try:
    from qhttp.qec_manager import QECManager
    from qhttp.grover_distributed import DistributedGrover
    import src.mcp_quantum_advanced as mcp_quantum_advanced
    QUANTUM_ADV_LOADED = True
except ImportError:
    QUANTUM_ADV_LOADED = False

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(asctime)s] %(message)s'))
logger = colorlog.getLogger('ArkheBoot')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ArkheKernel:
    def __init__(self):
        self.running = False
        self.simulation = None
        self.field = None
        self.mcp = None
        self.mcp_adv = None
        self.parallax_client = None
        self.grover = None
        self.qec = None
        self.stats = {'start_time': time.time()}

    async def boot(self, num_agents=150, parallax_client=None, controller_url=None):
        logger.info("🚀 Arkhe(n) OS Booting...")
        self.parallax_client = parallax_client
        if SharedFieldManager:
            self.field = SharedFieldManager()
            await self.field.initialize()

        if BIOGENESIS_LOADED:
            self.simulation = BioGenesisEngine(num_agents=num_agents)
            if create_mcp_server:
                self.mcp = create_mcp_server(self, self.parallax_client)

            if QUANTUM_ADV_LOADED and self.parallax_client:
                # Grover
                self.grover = DistributedGrover(self.parallax_client, controller_url)
                await self.grover.initialize()
                mcp_quantum_advanced.grover_engine = self.grover

                # QEC
                self.qec = QECManager(self.parallax_client)
                await self.qec.initialize()
                mcp_quantum_advanced.qec_manager = self.qec

                self.mcp_adv = mcp_quantum_advanced.mcp

        self.running = True
        return True

    async def simulation_loop(self):
        logger.info("🧠 Simulation Loop Started")
        while self.running:
            start_time = time.perf_counter()
            if self.simulation:
                try:
                    self.simulation.update(dt=0.1)
                    if self.field and hasattr(self.simulation, 'field'):
                        self.field.update_field(self.simulation.field.grid)
                except Exception as e:
                    logger.error(f"Sim error: {e}")
            await asyncio.sleep(max(0.01, 0.1 - (time.perf_counter() - start_time)))

    async def shutdown(self):
        self.running = False
        if self.field: await self.field.cleanup()
        if self.qec: await self.qec.stop()

kernel = ArkheKernel()
app = FastAPI(title="Arkhe(n) OS")

@app.get("/health")
async def health():
    return {"status": "healthy", "qec_active": kernel.parallax_client.qec_active if kernel.parallax_client else False}

async def main_async(args):
    client = None
    if args.mode == 'worker' and PARALLAX_LOADED:
        client = ParallaxNodeClient(args.node_id, args.controller, None, None)

    await kernel.boot(num_agents=args.agents, parallax_client=client, controller_url=args.controller)

    if client:
        client.simulation = kernel.simulation
        client.field = kernel.field
        await client.connect()

    asyncio.create_task(kernel.simulation_loop())

    if kernel.mcp:
        threading.Thread(target=lambda: kernel.mcp.run(transport="sse", port=8001), daemon=True).start()

    if kernel.mcp_adv:
        # Note: running multiple FastMCP in same process might conflict if not careful.
        # Here we run it on 8002.
        threading.Thread(target=lambda: kernel.mcp_adv.run(transport="sse", port=8002), daemon=True).start()

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'standalone'], default='standalone')
    parser.add_argument('--node-id', default='q1')
    parser.add_argument('--controller', default='http://parallax-controller:8080')
    parser.add_argument('--agents', type=int, default=150)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
