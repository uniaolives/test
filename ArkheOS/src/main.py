#!/usr/bin/env python3
"""
ARKHE(N) BOOTLOADER v2.0-PARALLAX
Suporte a modos: standalone | worker | controller
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import colorlog

# Importação dos módulos Arkhe
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
    except ImportError as e:
        BIOGENESIS_LOADED = False
        BioGenesisEngine = None
        SharedFieldManager = None
        create_mcp_server = None

# Importação Parallax
try:
    from parallax.node_client import ParallaxNodeClient
    PARALLAX_LOADED = True
except ImportError:
    PARALLAX_LOADED = False

# Configuração de logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white'}
))
logger = colorlog.getLogger('ArkheBoot')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ArkheKernel:
    def __init__(self):
        self.running = False
        self.simulation = None
        self.field = None
        self.mcp = None
        self.parallax_client = None
        self.stats = {'start_time': time.time(), 'updates': 0}

    async def boot(self, num_agents=150, parallax_client=None):
        logger.info("🚀 Arkhe(n) Kernel Booting...")
        self.parallax_client = parallax_client
        if SharedFieldManager:
            self.field = SharedFieldManager()
            await self.field.initialize()
        if BIOGENESIS_LOADED and BioGenesisEngine:
            self.simulation = BioGenesisEngine(num_agents=num_agents)
            if create_mcp_server:
                self.mcp = create_mcp_server(self, self.parallax_client)
        self.running = True
        return True

    async def simulation_loop(self):
        if not self.simulation: return
        while self.running:
            start_time = time.perf_counter()
            try:
                self.simulation.update(dt=0.1)
                self.stats['updates'] += 1
                if self.field and hasattr(self.simulation, 'field'):
                    self.field.update_field(self.simulation.field.grid)
            except Exception as e: logger.error(f"Sim error: {e}")
            await asyncio.sleep(max(0, 0.1 - (time.perf_counter() - start_time)))

    async def shutdown(self):
        self.running = False
        if self.field: await self.field.cleanup()
        logger.info("✅ Kernel shutdown complete.")

kernel = ArkheKernel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await kernel.boot()
    sim_task = asyncio.create_task(kernel.simulation_loop())
    yield
    kernel.running = False
    await sim_task
    await kernel.shutdown()

app = FastAPI(title="Arkhe(n) Core OS", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def root(): return "<h1>🧬 Arkhe(n) Core OS v2.0</h1><p>Status: OPERACIONAL</p>"

@app.get("/health")
async def health(): return {"status": "healthy", "uptime": time.time() - kernel.stats['start_time']}

def parse_args():
    parser = argparse.ArgumentParser(description='Arkhe(n) Core OS')
    parser.add_argument('--mode', choices=['standalone', 'worker', 'controller'], default='standalone')
    parser.add_argument('--node-id', default='auto')
    parser.add_argument('--controller', default='http://parallax-controller:8080')
    parser.add_argument('--agents', type=int, default=150)
    return parser.parse_args()

async def run_worker(args):
    logger.info(f"🖥️  Starting WORKER mode ({args.node_id})")
    if PARALLAX_LOADED:
        client = ParallaxNodeClient(args.node_id, args.controller, None, None) # Simulation/Field added after boot
        await kernel.boot(num_agents=args.agents, parallax_client=client)
        client.simulation = kernel.simulation
        client.field = kernel.field
        await client.connect()
        sim_task = asyncio.create_task(kernel.simulation_loop())
        if kernel.mcp:
            threading.Thread(target=lambda: kernel.mcp.run(transport="sse", port=8001), daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else: logger.error("Parallax not found")

async def run_controller():
    from parallax.controller import app as controller_app
    uvicorn.run(controller_app, host="0.0.0.0", port=8080)

async def run_standalone():
    await kernel.boot()
    if kernel.mcp:
        threading.Thread(target=lambda: kernel.mcp.run(transport="sse", port=8001), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'controller': asyncio.run(run_controller())
    elif args.mode == 'worker': asyncio.run(run_worker(args))
    else: asyncio.run(run_standalone())
