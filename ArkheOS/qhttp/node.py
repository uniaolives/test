# qhttp/node.py
import ctypes
import numpy as np
import logging
import os

logger = logging.getLogger("QHTTP.Node")

class QHTTPNode:
    def __init__(self, node_id, num_qubits=20, redis_url=None):
        self.node_id = node_id
        self.num_qubits = num_qubits
        self.redis_url = redis_url
        self.lib = None
        self.running = False

    async def start(self):
        logger.info(f"⚛️ Iniciando QHTTP Node {self.node_id}")
        lib_path = "/opt/arkhe/lib/libqhttp_dist.so"
        if os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                # self.lib.qhttp_dist_init(...)
                logger.info("✅ QHTTP Dist Lib carregada")
            except Exception as e:
                logger.error(f"Erro ao carregar libqhttp_dist.so: {e}")
        else:
            logger.warning(f"⚠️ libqhttp_dist.so não encontrada em {lib_path}")

        self.running = True

    async def stop(self):
        self.running = False

    def entangle_remote(self, local_agent, remote_rank, remote_agent, bell_type=0):
        if self.lib:
            # return self.lib.qhttp_entangle_remote(local_agent, remote_rank, remote_agent, bell_type)
            return True
        return False

    def collapse(self, agent_id):
        if self.lib:
            measured = ctypes.c_int()
            # self.lib.qhttp_collapse_remote(agent_id, ctypes.byref(measured))
            return measured.value
        return 0
