# formal/monitors/qnet_log_consumer.py
"""
Consumes logs from libqnet for runtime verification
Requires: libqnet message logging enabled
"""

import socket
import json

class QNetLogConsumer:
    def __init__(self, qnet_log_port: int = 9999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Handle case where port might be in use
        try:
            self.sock.bind(('0.0.0.0', qnet_log_port))
            self.sock.settimeout(1.0)
        except Exception as e:
            print(f"Warning: Could not bind to log port {qnet_log_port}: {e}")

    def receive_consensus_event(self) -> dict:
        """Receive consensus message logged by libqnet"""
        try:
            data, addr = self.sock.recvfrom(4096)
            return json.loads(data)
        except socket.timeout:
            return {}
        except Exception as e:
            print(f"Error receiving log: {e}")
            return {}
