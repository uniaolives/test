# parallax/qnet_interface.py (VERSÃO COMPLETA)
import ctypes
import os
from typing import Optional

# Load library
# Note: Path adjusted to point to ArkheOS/lib/libqnet.so
_lib_path = os.path.join(os.path.dirname(__file__), "../lib/libqnet.so")
try:
    _libqnet = ctypes.CDLL(_lib_path)
except OSError:
    # Fallback for development if lib is not yet compiled
    _libqnet = None

if _libqnet:
    # Define function signatures
    _libqnet.qnet_init.argtypes = [ctypes.c_char_p]
    _libqnet.qnet_init.restype = ctypes.c_int

    _libqnet.qnet_send.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libqnet.qnet_send.restype = ctypes.c_int

    _libqnet.qnet_recv.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libqnet.qnet_recv.restype = ctypes.c_int

    _libqnet.qnet_close.argtypes = []
    _libqnet.qnet_close.restype = None


class QNetError(Exception):
    """Exception raised for QNet errors"""
    pass


class QNet:
    """
    Python interface to DPDK-based ultra-low-latency networking.

    Usage:
        qnet = QNet()
        qnet.send(b"Hello, DPDK!")
        data = qnet.recv()
    """

    def __init__(self, pci_addr: str = "0000:01:00.0"):
        """Initialize DPDK networking"""
        self._initialized = False

        if _libqnet is None:
            raise QNetError(f"libqnet.so not found at {_lib_path}")

        ret = _libqnet.qnet_init(pci_addr.encode('utf-8'))
        if ret != 0:
            raise QNetError(f"Failed to initialize QNet (error {ret})")

        self._initialized = True

    def send(self, data: bytes) -> int:
        """
        Send data via DPDK.

        Returns:
            Number of bytes sent

        Raises:
            QNetError if send fails
        """
        if not self._initialized:
            raise QNetError("QNet not initialized")

        ret = _libqnet.qnet_send(data, len(data))
        if ret < 0:
            raise QNetError(f"Send failed (error {ret})")

        return ret

    def recv(self, max_len: int = 4096, timeout: Optional[float] = None) -> bytes:
        """
        Receive data (non-blocking).

        Returns:
            Received bytes (empty if nothing available)

        Note:
            timeout parameter is ignored (always non-blocking)
        """
        if not self._initialized:
            raise QNetError("QNet not initialized")

        buffer = ctypes.create_string_buffer(max_len)
        ret = _libqnet.qnet_recv(buffer, max_len)

        if ret < 0:
            raise QNetError(f"Recv failed (error {ret})")

        if ret == 0:
            return b""

        return buffer.raw[:ret]

    def close(self):
        """Cleanup DPDK resources"""
        if self._initialized:
            _libqnet.qnet_close()
            self._initialized = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
