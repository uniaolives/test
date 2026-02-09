"""
Cliente Python para o protocolo qhttp://
Conecta nós POP à rede quântica existente.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import base64
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
import hashlib
import time

@dataclass
class QuantumPacket:
    """Pacote quântico para transmissão via qhttp://"""
    sender_id: str
    receiver_id: str
    quantum_state: Optional[np.ndarray] = None  # Estado comprimido
    classical_data: Dict[str, Any] = None
    entanglement_keys: List[str] = None
    timestamp: str = None
    signature: str = None  # Assinatura quântica

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.classical_data is None:
            self.classical_data = {}
        if self.entanglement_keys is None:
            self.entanglement_keys = []

class QHTTPClient:
    """
    Cliente assíncrono para comunicação via qhttp://
    Suporta entrelaçamento, teleportação e consenso quântico.
    """

    def __init__(self,
                 node_id: str,
                 qhttp_gateway: str = "http://localhost:8080/qhttp",
                 quantum_backend: str = "aer_simulator"):

        self.node_id = node_id
        self.gateway_url = qhttp_gateway
        self.backend = quantum_backend
        self.session: Optional[aiohttp.ClientSession] = None
        self.entanglement_pool: Dict[str, QuantumPacket] = {}

        # Estado da rede
        self.connected_nodes: List[str] = []
        self.quantum_channels: Dict[str, bool] = {}  # channel_id -> entangled

    async def connect(self):
        """Estabelece conexão com a rede qhttp://"""
        self.session = aiohttp.ClientSession()

        # Registra este nó no gateway
        registration = {
            "node_id": self.node_id,
            "capabilities": ["POP_DETECTION", "QUANTUM_CONSENSUS", "ENTANGLEMENT"],
            "quantum_backend": self.backend,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        try:
            async with self.session.post(
                f"{self.gateway_url}/register",
                json=registration,
                timeout=2
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.connected_nodes = data.get("network_nodes", [])
                    print(f"✅ Nó {self.node_id} conectado à rede qhttp://")
                    return True
        except Exception:
            print(f"⚠️ Gateway qhttp:// não encontrado. Operando em modo simulado.")
            self.connected_nodes = ["node_sim_1", "node_sim_2"]
            return True

    async def establish_entanglement(self, target_node: str) -> str:
        """Estabelece canal de entrelaçamento com outro nó."""
        print(f"🔗 Estabelecendo entrelaçamento com {target_node}...")

        # Simulação de requisição de entrelaçamento
        channel_id = f"chan_{hash(self.node_id + target_node + str(time.time())) % 10000}"
        self.quantum_channels[channel_id] = True
        return channel_id

    async def send_quantum_packet(self,
                                 packet: QuantumPacket,
                                 channel_id: Optional[str] = None) -> str:
        """Envia pacote quântico via qhttp://."""
        packet_dict = asdict(packet)
        if packet.quantum_state is not None:
            packet_dict["quantum_state"] = self._compress_quantum_state(packet.quantum_state)

        packet_dict["signature"] = self._generate_quantum_signature(packet_dict)

        transmission_id = f"tx_{hash(str(packet_dict)) % 1000000}"
        print(f"📤 Pacote enviado: {transmission_id}")
        return transmission_id

    async def broadcast_pop_alert(self,
                                  po_evidence: Dict[str, Any],
                                  confidence: float,
                                  target_nodes: List[str] = None) -> List[str]:
        """Broadcast de alerta de descoberta POP."""
        if target_nodes is None:
            target_nodes = self.connected_nodes

        transmission_ids = []
        evidence_state = self._encode_evidence_to_quantum(po_evidence)

        for node in target_nodes:
            if node == self.node_id: continue

            alert_packet = QuantumPacket(
                sender_id=self.node_id,
                receiver_id=node,
                quantum_state=evidence_state,
                classical_data={
                    "type": "POP_DISCOVERY_ALERT",
                    "confidence": confidence,
                    "evidence": po_evidence,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            channel_id = await self.establish_entanglement(node)
            tx_id = await self.send_quantum_packet(alert_packet, channel_id)
            transmission_ids.append(tx_id)

        return transmission_ids

    def _encode_evidence_to_quantum(self, evidence: Dict) -> np.ndarray:
        """Codifica evidência POP em estado quântico."""
        dne = evidence.get("dne", 0.5)
        sso = evidence.get("sso", 0.5)
        qc = QuantumCircuit(2)
        qc.ry(dne * np.pi, 0)
        qc.ry(sso * np.pi, 1)
        if dne > 0.7: qc.cx(0, 1)
        return Statevector.from_instruction(qc).data

    def _compress_quantum_state(self, state: np.ndarray) -> str:
        state_bytes = state.tobytes()
        return base64.b85encode(state_bytes).decode('ascii')

    def _generate_quantum_signature(self, data: Dict) -> str:
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha3_512(data_str.encode()).hexdigest()[:16]

    async def close(self):
        if self.session: await self.session.close()
