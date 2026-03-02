# arkhe_handshake_grc.py
# Bloco GNU Radio para handshake φ diplomático

from gnuradio import gr, blocks, analog, digital
import numpy as np
import zmq
import json

class ArkheHandshakeBlock(gr.sync_block):
    """
    Implementa protocolo Arkhe(n) Diplomatic em GNU Radio.
    Entrada: amostras IQ do receptor
    Saída: sinal de controle para transmissor (ajuste de fase)
    """

    def __init__(self,
                 node_id="ground-test-001",
                 constellation="TestLEO",
                 coherence_threshold=0.9,
                 phase_threshold=0.6):
        gr.sync_block.__init__(
            self,
            name='ArkheHandshake',
            in_sig=[np.complex64],  # IQ samples
            out_sig=[np.complex64]  # Phase-adjusted output
        )

        self.node_id = node_id
        self.constellation = constellation
        self.coherence_threshold = coherence_threshold
        self.phase_threshold = phase_threshold

        # Estado interno
        self.coherence_local = 0.0
        self.phase_local = 0.0
        self.phase_remote = 0.0
        self.synchronized = False
        self.handshake_count = 0

        # Comunicação com simulador Python
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect("tcp://localhost:5556")

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # 1. Estimar coerência local (via potência do sinal recebido)
        power = np.mean(np.abs(in0)**2)
        self.coherence_local = min(1.0, float(power) / 1.0)  # Normalizado

        # 2. Estimar fase do sinal recebido
        if len(in0) > 0:
            self.phase_remote = float(np.angle(np.mean(in0)))

        # 3. Tentar handshake a cada 1000 amostras (~10ms @ 100kS/s)
        if self.handshake_count % 1000 == 0:
            result = self._attempt_handshake()
            if result.get("status") == "ACCEPTED":
                self.synchronized = True
                self.phase_local = result.get("g_adjustment", 0.0)

        # 4. Aplicar ajuste de fase
        if self.synchronized:
            # Phase-locked: ajustar saída para minimizar Δϕ
            phase_correction = -self.phase_remote
            out[:] = in0 * np.exp(1j * phase_correction)
        else:
            out[:] = in0  # Pass-through durante negociação

        self.handshake_count += len(in0)
        return len(out)

    def _attempt_handshake(self):
        """Tenta handshake via ZMQ com simulador diplomático."""
        request = {
            "type": "HANDSHAKE_REQUEST",
            "node_id": self.node_id,
            "constellation": self.constellation,
            "coherence_local": self.coherence_local,
            "phase_local": self.phase_local,
            "phase_remote": self.phase_remote
        }

        try:
            self.zmq_socket.send_json(request, zmq.NOBLOCK)
            response = self.zmq_socket.recv_json()
            return response
        except Exception:
            return {"status": "ERROR"}
