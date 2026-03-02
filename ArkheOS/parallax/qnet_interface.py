# parallax/qnet_interface.py
import ctypes
import os
import logging

logger = logging.getLogger("Parallax.QNet")

class QNetInterface:
    """
    Interface Python para a biblioteca de rede DPDK (libqnet).
    Permite comunicação de ultra-baixa latência (<5us).
    """
    def __init__(self):
        self.lib = None
        self._init_lib()

    def _init_lib(self):
        lib_path = "/opt/arkhe/lib/libqnet.so"
        if os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                logger.info("✅ libqnet.so carregada.")
            except Exception as e:
                logger.error(f"Erro ao carregar libqnet.so: {e}")
        else:
            logger.warning("⚠️ libqnet.so não encontrada.")

    def initialize(self, args: list):
        if not self.lib: return False
        # Converte args para char**
        ArgArray = ctypes.c_char_p * len(args)
        c_args = ArgArray(*[arg.encode() for arg in args])
        ret = self.lib.qnet_init(len(args), c_args)
        return ret == 0

    def send(self, port_id: int, data: bytes):
        if not self.lib: return False
        return self.lib.qnet_send_packet(port_id, data, len(data)) == 0

    def receive(self, port_id: int, max_len: int = 1500) -> bytes:
        if not self.lib: return b""
        buf = ctypes.create_string_buffer(max_len)
        length = self.lib.qnet_recv_packet(port_id, buf, max_len)
        if length > 0:
            return buf.raw[:length]
        return b""
