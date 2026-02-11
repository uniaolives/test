"""
QEC Manager – Ativa/desativa ciclos de correção de erros nos nós.
"""

import asyncio
import ctypes
import logging
import os
import numpy as np
from typing import Optional

logger = logging.getLogger("QEC.Manager")

class QECManager:
    def __init__(self, node_client=None):
        self.node_client = node_client
        self.libqec = None
        self.active = False
        self.cycle_interval = 0.1  # segundos
        self.task = None

    async def initialize(self):
        """Carrega libqec.so (kernels de Surface Code)."""
        lib_path = "/opt/arkhe/lib/libqec.so"
        if not os.path.exists(lib_path):
            logger.error(f"libqec.so não encontrada em {lib_path}. Q‑Shield desativado.")
            return False

        try:
            self.libqec = ctypes.CDLL(lib_path)
            self.libqec.qec_cycle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.libqec.qec_cycle.restype = ctypes.c_int
            logger.info("✅ libqec.so carregada. Surface Code disponível.")
            return True
        except Exception as e:
            logger.error(f"Falha ao carregar libqec.so: {e}")
            return False

    async def start(self):
        """Inicia loop de correção contínua."""
        if self.active:
            return
        self.active = True
        self.task = asyncio.create_task(self._correction_loop())
        logger.info("🛡️ Q‑Shield ATIVADO")

    async def stop(self):
        self.active = False
        if self.task:
            self.task.cancel()
        logger.info("🛡️ Q‑Shield desativado.")

    async def _correction_loop(self):
        while self.active:
            try:
                if self.node_client and self.node_client.d_agents:
                    n_agents = self.node_client.num_agents
                    grid_dim = int(np.sqrt(n_agents))
                    ret = self.libqec.qec_cycle(
                        self.node_client.d_agents, n_agents, grid_dim
                    )
                    if ret != 0:
                        logger.warning("Ciclo QEC falhou")
                await asyncio.sleep(self.cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no ciclo QEC: {e}")
