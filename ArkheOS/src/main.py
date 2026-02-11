#!/usr/bin/env python3
"""
ARKHE(N) BOOTLOADER v1.0
Inicializa todos os componentes do sistema operacional biológico.
"""

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

# Configuração de logging colorido
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger('ArkheBoot')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
        logger.error(f"❌ Falha ao carregar módulos: {e}")
        BIOGENESIS_LOADED = False
        BioGenesisEngine = None
        SharedFieldManager = None
        create_mcp_server = None

class ArkheSystem:
    """Sistema principal Arkhe(n) OS."""

    def __init__(self):
        self.running = False
        self.simulation = None
        self.shared_field = None
        self.mcp = None
        self.stats = {
            'start_time': time.time(),
            'updates': 0
        }

    async def initialize(self):
        """Inicializa todos os componentes do sistema."""
        logger.info("🚀 Inicializando Arkhe(n) Core OS v1.0")

        if SharedFieldManager:
            self.shared_field = SharedFieldManager()
            await self.shared_field.initialize()

        if BIOGENESIS_LOADED and BioGenesisEngine:
            self.simulation = BioGenesisEngine(num_agents=150)
            logger.info(f"✅ Bio-Gênese carregado: {len(self.simulation.agents)} agentes")

            if create_mcp_server:
                self.mcp = create_mcp_server(self)

        self.running = True
        return True

    async def simulation_loop(self):
        """Loop principal da simulação."""
        if not self.simulation or not self.running:
            return

        logger.info("🧠 Iniciando loop de simulação (10Hz)...")
        while self.running:
            start_time = time.perf_counter()
            try:
                self.simulation.update(dt=0.1)
                self.stats['updates'] += 1
                if self.shared_field and hasattr(self.simulation, 'field'):
                    self.shared_field.update_field(self.simulation.field.grid)
            except Exception as e:
                logger.error(f"Erro no loop de simulação: {e}")

            elapsed = time.perf_counter() - start_time
            await asyncio.sleep(max(0, 0.1 - elapsed))

    async def shutdown(self):
        self.running = False
        if self.shared_field:
            await self.shared_field.cleanup()
        logger.info("✅ Sistema Arkhe(n) desligado.")

arkhe_system = ArkheSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await arkhe_system.initialize()
    simulation_task = asyncio.create_task(arkhe_system.simulation_loop())
    yield
    arkhe_system.running = False
    await simulation_task
    await arkhe_system.shutdown()

app = FastAPI(title="Arkhe(n) Core OS", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body style="font-family: monospace; background: #0a0a0a; color: #0af; padding: 40px;">
            <h1>🧬 Arkhe(n) Core OS v1.0</h1>
            <p>Status: <span style="color: #0f0;">OPERACIONAL</span></p>
            <p>Servidor MCP ativo na porta 8001 (SSE)</p>
            <p><a href="/health" style="color: #0ff;">Health Check</a> | <a href="/docs" style="color: #0ff;">API Docs</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agents": len(arkhe_system.simulation.agents) if arkhe_system.simulation else 0,
        "uptime": time.time() - arkhe_system.stats.get('start_time', 0)
    }

def run_mcp():
    if arkhe_system.mcp:
        logger.info("🔌 Iniciando Servidor MCP na porta 8001...")
        arkhe_system.mcp.run(transport="sse", port=8001)

def main():
    # Aguarda inicialização básica para ter o objeto MCP
    # Na verdade, o initialize roda no lifespan do FastAPI,
    # então precisamos que o MCP rode depois ou de forma resiliente.

    # Vamos rodar o inicializador aqui fora também ou garantir que run_mcp aguarda.
    def mcp_bootstrap():
        # Aguarda até que arkhe_system.mcp esteja disponível
        for _ in range(10):
            if arkhe_system.mcp:
                run_mcp()
                return
            time.sleep(1)
        logger.error("❌ Servidor MCP não pôde ser iniciado: tempo esgotado.")

    mcp_thread = threading.Thread(target=mcp_bootstrap, daemon=True)
    mcp_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
