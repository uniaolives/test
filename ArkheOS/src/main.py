# src/main.py
"""
ARKHE(N) OS v2.0 - Unified Boot Sequence
Integra Bio-Gênese + Quantum + Framework Arkhe Hexagonal
"""

import asyncio
import argparse
import logging
import os
import numpy as np
from typing import Optional

# Imports Arkhe
from src.arkhe.distributed_hexagonal_engine import DistributedHexagonalEngine
from src.arkhe.distributed_light_cone import DistributedCognitiveLightCone
from src.arkhe.neuro_arkhe_bridge import NeuroArkheBridge
from parallax.worker import ParallaxWorker
from qhttp.node import QHTTPNode
from qhttp.qec.shield import QShieldManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARKHE(N)")


class ArkheUnifiedNode:
    """
    Nó unificado executando todas as camadas:
    1. Parallax Worker (migração clássica)
    2. QHTTP Node (rede quântica)
    3. Q-Shield (QEC)
    4. Arkhe Hexagonal Engine (geometria consciente)
    5. Cognitive Light Cone (inteligência)
    6. Neuro-Arkhe Bridge (interface)
    """

    def __init__(self, node_id: str, redis_url: str):
        self.node_id = node_id
        self.redis_url = redis_url

        # Componentes
        self.parallax_worker: Optional[ParallaxWorker] = None
        self.qhttp_node: Optional[QHTTPNode] = None
        self.qshield: Optional[QShieldManager] = None
        self.arkhe_engine: Optional[DistributedHexagonalEngine] = None
        self.light_cone: Optional[DistributedCognitiveLightCone] = None
        self.neuro_bridge: Optional[NeuroArkheBridge] = None

        self._tasks = []

    async def initialize(self):
        """Boot sequence completo"""
        logger.info(f"🌌 Inicializando ARKHE(N) Node {self.node_id}")

        # 1. Parallax Worker
        logger.info("  [1/6] Parallax Worker...")
        self.parallax_worker = ParallaxWorker(
            node_id=self.node_id,
            controller_url=os.getenv("PARALLAX_CONTROLLER_URL", "http://controller:8080"),
            redis_url=self.redis_url
        )
        await self.parallax_worker.start()

        # 2. QHTTP Node
        logger.info("  [2/6] QHTTP Quantum Network...")
        self.qhttp_node = QHTTPNode(
            node_id=self.node_id,
            num_qubits=20,
            redis_url=self.redis_url
        )
        await self.qhttp_node.start()

        # 3. Q-Shield
        logger.info("  [3/6] Q-Shield Surface Code...")
        self.qshield = QShieldManager(
            node_id=self.node_id,
            code_distance=7,  # Distance 7 para maior proteção
            redis_url=self.redis_url
        )
        await self.qshield.start()

        # 4. Arkhe Hexagonal Engine
        logger.info("  [4/6] Arkhe Hexagonal Geometry...")
        self.arkhe_engine = DistributedHexagonalEngine(
            node_id=self.node_id,
            redis_url=self.redis_url,
            total_nodes=3,
            beta=2.0
        )
        await self.arkhe_engine.initialize()

        # 5. Cognitive Light Cone
        logger.info("  [5/6] Cognitive Light Cone Engine...")
        self.light_cone = DistributedCognitiveLightCone(
            node_id=self.node_id,
            redis_url=self.redis_url,
            local_qubits=10,
            num_nodes=3
        )

        # 6. Neuro-Arkhe Bridge
        logger.info("  [6/6] Neuro-Arkhe Consciousness Bridge...")
        self.neuro_bridge = NeuroArkheBridge(
            node_id=self.node_id,
            redis_url=self.redis_url,
            arkhe_engine=self.arkhe_engine
        )

        logger.info(f"✅ ARKHE(N) Node {self.node_id} ONLINE")
        logger.info(f"   Arkhe Coherence: {self.arkhe_engine.local_state.coherence:.3f}")

    async def run_consciousness_loop(self):
        """
        Loop principal de consciência:
        EEG → Arkhe → Quantum → Visualização
        """
        logger.info(f"🧠 Iniciando Consciousness Loop ({self.node_id})")

        while True:
            try:
                # 1. Simula contexto MCP (em produção, vem do gateway)
                simulated_query = "quantum entangle agents across nodes"

                # 2. EEG simulado baseado em query
                eeg_data = await self.neuro_bridge.simulate_eeg_from_mcp_context(
                    simulated_query
                )

                # 3. Sincroniza Arkhe com estado quântico
                await self.arkhe_engine.sync_with_quantum_state(self.node_id)

                # 4. Evolui consciência (Arkhe + restrições)
                await self.neuro_bridge.consciousness_to_arkhe_evolution(eeg_data)

                # 5. Mede inteligência (volume do cone de luz)
                # (Requer estado do agente - simplificado aqui)

                # 6. Publica métricas
                await self._publish_metrics()

                await asyncio.sleep(0.2)  # 5 Hz

            except Exception as e:
                logger.error(f"Consciousness loop error: {e}")
                await asyncio.sleep(1.0)

    async def _publish_metrics(self):
        """Publica métricas consolidadas para Prometheus"""
        import redis.asyncio as aioredis
        redis = await aioredis.from_url(self.redis_url)

        await redis.hset(f"metrics:{self.node_id}", mapping={
            'arkhe_coherence': str(self.arkhe_engine.local_state.coherence),
            'dominant_component': str(int(np.argmax(self.arkhe_engine.local_state.components))),
            'qec_active': "true" if self.qshield.is_active else "false",
            'num_bio_agents': str(self.parallax_worker.num_agents if self.parallax_worker else 0)
        })

        await redis.close()

    async def run(self):
        """Executa node indefinidamente"""
        await self.initialize()

        # Inicia loops paralelos
        self._tasks = [
            asyncio.create_task(self.run_consciousness_loop()),
            # Outros loops podem ser adicionados
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            logger.info("Shutdown solicitado")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"🔻 Desligando {self.node_id}...")

        for task in self._tasks:
            task.cancel()

        if self.qshield:
            await self.qshield.stop()
        if self.qhttp_node:
            await self.qhttp_node.stop()
        if self.parallax_worker:
            await self.parallax_worker.stop()

        logger.info("✅ Shutdown completo")


async def main():
    parser = argparse.ArgumentParser(description="ARKHE(N) OS v2.0")
    parser.add_argument('--mode', choices=['worker', 'gateway', 'controller'], required=True)
    parser.add_argument('--node-id', required=True)
    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")

    if args.mode == 'worker':
        node = ArkheUnifiedNode(
            node_id=args.node_id,
            redis_url=redis_url
        )
        await node.run()
    elif args.mode == 'controller':
        from parallax.controller import main as controller_main
        await controller_main()
    elif args.mode == 'gateway':
        from qhttp.gateway import main as gateway_main
        await gateway_main()


if __name__ == "__main__":
    asyncio.run(main())
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
    from .arkhe.cortex_memory import CortexMemory
    from .open_context_mcp import create_open_context_server
    BIOGENESIS_LOADED = True
except ImportError:
    try:
        from particle_system import BioGenesisEngine
        from shared_memory import SharedFieldManager
        from mcp_server import create_mcp_server
        from arkhe.cortex_memory import CortexMemory
        from open_context_mcp import create_open_context_server
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
        self.open_context_mcp = None
        self.cortex = None
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

            # Inicializa Córtex e Open Context MCP
            if CortexMemory:
                self.cortex = CortexMemory()
                logger.info("🧠 Córtex (Vector DB) inicializado.")
                if create_open_context_server:
                    self.open_context_mcp = create_open_context_server(self.cortex)
                    logger.info("🌐 Servidor Open Context MCP configurado.")

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
        logger.info("🔌 Iniciando Servidor ArkheOS MCP na porta 8001...")
        # Rodar em thread separada para não bloquear
        # Note: FastMCP.run binds to 0.0.0.0 by default, check if we need 127.0.0.1
        threading.Thread(target=lambda: arkhe_system.mcp.run(transport="sse", port=8001), daemon=True).start()

def run_open_context_mcp():
    if arkhe_system.open_context_mcp:
        logger.info("🌐 Iniciando Servidor Open Context MCP na porta 8002 (Localhost)...")
        # Rodar em thread separada
        threading.Thread(target=lambda: arkhe_system.open_context_mcp.run(transport="sse"), daemon=True).start()

def main():
    # Aguarda inicialização básica para ter os objetos MCP
    def mcp_bootstrap():
        # Aguarda até que os servidores MCP estejam disponíveis
        for _ in range(10):
            if arkhe_system.mcp or arkhe_system.open_context_mcp:
                run_mcp()
                run_open_context_mcp()
                return
            time.sleep(1)
        logger.error("❌ Servidores MCP não puderam ser iniciados: tempo esgotado.")

    mcp_thread = threading.Thread(target=mcp_bootstrap, daemon=True)
    mcp_thread.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Shutdown solicitado pelo usuário.")

if __name__ == "__main__":
    main()
