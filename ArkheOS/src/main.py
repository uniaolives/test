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
    log_colors={
        'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger('ArkheBoot')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ArkheKernel:
    """Núcleo do sistema operacional Arkhe(n)"""
    def __init__(self):
        self.running = False
        self.simulation = None
        self.field = None
        self.mcp = None
        self.stats = {'start_time': time.time(), 'updates': 0}

    async def boot(self, num_agents=150):
        logger.info("🚀 Arkhe(n) Kernel Booting...")
        if SharedFieldManager:
            self.field = SharedFieldManager()
            await self.field.initialize()

        if BIOGENESIS_LOADED and BioGenesisEngine:
            self.simulation = BioGenesisEngine(num_agents=num_agents)
            logger.info(f"✅ Bio-Genesis v3.0 active with {len(self.simulation.agents)} agents")
            if create_mcp_server:
                self.mcp = create_mcp_server(self)

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
            except Exception as e:
                logger.error(f"Sim error: {e}")
            elapsed = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.1 - elapsed))

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
async def root():
    return "<h1>🧬 Arkhe(n) Core OS v2.0</h1><p>Status: OPERACIONAL</p>"

@app.get("/health")
async def health():
    return {"status": "healthy", "uptime": time.time() - kernel.stats['start_time']}

def parse_args():
    parser = argparse.ArgumentParser(description='Arkhe(n) Core OS')
    parser.add_argument('--mode', choices=['standalone', 'worker', 'controller'],
                       default='standalone', help='Operating mode')
    parser.add_argument('--node-id', default='auto', help='Node ID (worker mode)')
    parser.add_argument('--controller', default='http://parallax-controller:8080',
                       help='Controller URL (worker mode)')
    parser.add_argument('--agents', type=int, default=150, help='Initial population')
    return parser.parse_args()

async def run_worker(args):
    logger.info(f"🖥️  Starting WORKER mode (node {args.node_id})")
    await kernel.boot(num_agents=args.agents)

    if PARALLAX_LOADED:
        client = ParallaxNodeClient(
            node_id=args.node_id,
            controller_url=args.controller,
            simulation=kernel.simulation,
            field=kernel.field
        )
        await client.connect()
        sim_task = asyncio.create_task(kernel.simulation_loop())

        # Start local status server in thread
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        server_task = loop.create_task(server.serve())

        # MCP loop (FastMCP run blocks, so we run it in a thread if possible)
        if kernel.mcp:
            def run_mcp_local():
                kernel.mcp.run(transport="sse", port=8001)
            threading.Thread(target=run_mcp_local, daemon=True).start()

        try:
            while client.running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await client.disconnect()
            kernel.running = False
            await sim_task
            await kernel.shutdown()
    else:
        logger.error("❌ Parallax modules not found. Cannot start worker.")

async def run_controller():
    logger.info("🎛️  Starting CONTROLLER mode")
    from parallax.controller import app as controller_app
    config = uvicorn.Config(controller_app, host="0.0.0.0", port=8080)
    server = uvicorn.Server(config)
    await server.serve()

async def run_standalone():
    logger.info("🧬 Starting STANDALONE mode")
    if kernel.mcp:
        def run_mcp_local():
            kernel.mcp.run(transport="sse", port=8001)
        threading.Thread(target=run_mcp_local, daemon=True).start()

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

async def entrypoint():
    args = parse_args()
    if args.mode == 'controller':
        await run_controller()
    elif args.mode == 'worker':
        await run_worker(args)
    else:
        await run_standalone()

if __name__ == "__main__":
    try:
        asyncio.run(entrypoint())
    except KeyboardInterrupt:
        pass
