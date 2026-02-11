"""
Grover Distribuído – Busca global de agentes por genoma.
"""

import asyncio
import ctypes
import logging
import os
import numpy as np
import aiohttp
from typing import Optional, Dict, Any

logger = logging.getLogger("Grover.Distributed")

class DistributedGrover:
    def __init__(self, node_client, controller_url):
        self.node_client = node_client
        self.controller_url = controller_url
        self.libgrover = None

    async def initialize(self):
        lib_path = "/opt/arkhe/lib/libgrover.so"
        if not os.path.exists(lib_path):
            logger.error(f"libgrover.so não encontrada em {lib_path}. Grover desativado.")
            return False

        try:
            self.libgrover = ctypes.CDLL(lib_path)
            self.libgrover.grover_iteration.argtypes = [
                ctypes.c_void_p,  # d_agents
                ctypes.c_int,     # n (local)
                ctypes.c_int,     # target_genome
                ctypes.c_void_p,  # ncclComm_t
                ctypes.c_int,     # rank
                ctypes.c_int      # world_size
            ]
            self.libgrover.grover_iteration.restype = ctypes.c_int
            logger.info("✅ libgrover.so carregada.")
            return True
        except Exception as e:
            logger.error(f"Falha ao carregar libgrover.so: {e}")
            return False

    async def run_search(self, target_genome: int) -> Optional[Dict[str, Any]]:
        """Executa Grover distribuído e retorna o agente encontrado."""
        if not self.node_client or not self.node_client.qhttp_dist_lib: # Check if distributed quantum ready
            logger.error("Quantum distribuído não inicializado.")
            return None

        # Obter número total de agentes no cluster
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.controller_url}/health") as resp:
                    data = await resp.json()
                    total_agents = data.get('total_agents', 150)
        except:
            total_agents = 150

        iterations = int((np.pi / 4) * np.sqrt(total_agents))
        logger.info(f"🔍 Grover: {iterations} iterações para genoma {target_genome}")

        for i in range(iterations):
            # In a real scenario, we'd need access to the NCCL comm pointer from the node client
            # Here we assume the node client has it or the lib handles it
            ret = self.libgrover.grover_iteration(
                self.node_client.d_agents,
                self.node_client.num_agents,
                target_genome,
                None, # ncclComm_t stub
                self.node_client.nccl_rank,
                self.node_client.world_size
            )
            if ret != 0:
                logger.error(f"Falha na iteração {i}")
                return None

        # Medição
        measured = await self.node_client.measure_all_agents()
        return {"agent_id": measured, "genome": target_genome}
