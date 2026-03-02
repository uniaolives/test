"""
qci_handshake.py
Specification of the Quantum-Classical Interface (QCI) Handshake.
Syncs the Instaweb buffer with the quantum teleportation gates.
"""

import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QCI-Handshake")

class QCI_Controller:
    def __init__(self, node_id):
        self.node_id = node_id
        self.classical_buffer = [] # (msg_id, arrival_time, bell_data)
        self.qubit_buffer = {}     # (qubit_id) -> ready_time

    def on_classical_arrival(self, msg_id, bell_data):
        arrival_time = time.time_ns()
        self.classical_buffer.append((msg_id, arrival_time, bell_data))
        logger.info(f"[{self.node_id}] Classical message {msg_id} stored in buffer at {arrival_time}ns")
        self.process_handshake()

    def on_qubit_arrival(self, qubit_id):
        ready_time = time.time_ns()
        self.qubit_buffer[qubit_id] = ready_time
        logger.info(f"[{self.node_id}] Qubit {qubit_id} ready in lab at {ready_time}ns")
        self.process_handshake()

    def process_handshake(self):
        """Matches classical results with arrived qubits."""
        matched = []
        for i, (msg_id, t_msg, bell) in enumerate(self.classical_buffer):
            # Qhttp logic: message ID matches qubit sequence
            if msg_id in self.qubit_buffer:
                t_qubit = self.qubit_buffer[msg_id]

                # Verification of coherence window (T2 approx 100us)
                deadline = t_qubit + 100_000
                current = time.time_ns()

                if current < deadline:
                    logger.info(f"✅ [SYNC] Teleportation Success for {msg_id}!")
                    logger.info(f"   • Delta: {(current - t_qubit)/1000:.2f}us")
                    matched.append(i)
                    del self.qubit_buffer[msg_id]
                else:
                    logger.warning(f"❌ [FAIL] Decoherence for {msg_id}! Gate applied too late.")
                    matched.append(i)
                    del self.qubit_buffer[msg_id]

        # Cleanup buffer
        for index in sorted(matched, reverse=True):
            self.classical_buffer.pop(index)

if __name__ == "__main__":
    controller = QCI_Controller("Node_Bob")

    # Simulating synchronization
    controller.on_qubit_arrival("Q-001")
    time.sleep(0.00005) # 50us delay
    controller.on_classical_arrival("Q-001", "01") # Pauli-X correction
