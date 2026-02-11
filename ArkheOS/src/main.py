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
