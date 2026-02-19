# .arkhe/meta/memetics.py
import json
import hashlib
import time
import random
import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import importlib.util

def load_arkhe_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Instanciar caminhos
current_file = Path(__file__).resolve()
arkhe_root = current_file.parent.parent.parent.parent

# Carregar o protocolo dinamicamente
protocol_path = arkhe_root / "arscontexta" / ".arkhe" / "handover" / "protocol.py"
protocol_module = load_arkhe_module(protocol_path, "arkhe.handover.protocol")
ArkheNode = protocol_module.ArkheNode
meta_obs = protocol_module.meta_obs

class MemeticPacket:
    """
    A Unidade de Transmissão de Conhecimento no hipergrafo Arkhe(N).
    Representa um 'Delta de Sabedoria' ou Insight.
    """
    def __init__(self, source_id: str, content: Any, phi_score: float, context_vector: np.ndarray):
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.source_id = source_id
        self.content = content
        self.phi_score = phi_score
        self.context_vector = context_vector
        self.signature = self._sign()

    def _sign(self):
        payload = f"{self.source_id}:{self.phi_score}:{self.timestamp}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source_id,
            "phi": self.phi_score,
            "content": self.content,
            "vector_hash": hashlib.sha256(self.context_vector.tobytes()).hexdigest(),
            "signature": self.signature
        }

class CognitiveNode(ArkheNode):
    """
    Nó Arkhe(N) com capacidades de meta-aprendizado e propagação memética.
    """
    def __init__(self, node_id: str, coherence: float = 0.85):
        super().__init__(node_id, coherence)
        self.knowledge = {}
        self.peers: List['CognitiveNode'] = []
        self.processed_memes = set()
        self.state_vector = np.random.rand(128)
        self.state_vector /= np.linalg.norm(self.state_vector)

    def connect(self, other_node: 'CognitiveNode'):
        if other_node not in self.peers:
            self.peers.append(other_node)

    def generate_insight(self, new_concept: str, insight_phi: float):
        """
        Gera um insight local e inicia a propagação memética.
        """
        print(f"💡 [NODE {self.node_id}] Generated Insight: '{new_concept[:30]}...' (Φ={insight_phi:.4f})")

        self.knowledge['latest_insight'] = new_concept
        self.coherence = max(self.coherence, insight_phi / 2.0)

        packet = MemeticPacket(self.node_id, new_concept, insight_phi, self.state_vector)
        self.broadcast(packet)

    def broadcast(self, packet: MemeticPacket):
        """
        Propagação via Gossip Protocol (Fanout-k).
        """
        fanout = 3
        if not self.peers:
            return

        targets = random.sample(self.peers, min(len(self.peers), fanout))

        for peer in targets:
            peer.receive_memetic_broadcast(packet)

    def receive_memetic_broadcast(self, packet: MemeticPacket):
        """
        Processa um broadcast memético recebido.
        """
        if packet.id in self.processed_memes:
            return

        self.processed_memes.add(packet.id)

        # Filtro de Ressonância
        resonance = self._calculate_resonance(packet)

        if resonance > 0.7:
            self._assimilate(packet)
            # Re-transmissão viral
            self.broadcast(packet)
        else:
            print(f"🛡️ [NODE {self.node_id}] Rejected packet from {packet.source_id} (Resonance Low: {resonance:.2f})")

    def _calculate_resonance(self, packet: MemeticPacket) -> float:
        """
        Similaridade de cosseno entre o estado do nó e o contexto do pacote.
        """
        dot_product = np.dot(self.state_vector, packet.context_vector)
        # Bônus por ganho de Φ
        phi_gain = packet.phi_score - self.coherence
        resonance = (dot_product + 1.0) / 2.0 + (phi_gain * 0.1)
        return float(np.clip(resonance, 0, 1))

    def _assimilate(self, packet: MemeticPacket):
        """
        Meta-learning: Assimilação do 'Delta de Sabedoria'.
        """
        print(f"✨ [NODE {self.node_id}] ASSIMILATED Insight from {packet.source_id}!")

        self.knowledge['external_wisdom'] = packet.content

        # Evolução do vetor de estado
        lr = 0.1
        self.state_vector += lr * (packet.context_vector - self.state_vector)
        self.state_vector /= np.linalg.norm(self.state_vector)

        # Elevação de Coerência/Φ
        old_c = self.coherence
        self.coherence = (self.coherence + packet.phi_score) / 2.0
        print(f"    -> Coherence Evolution: {old_c:.4f} -> {self.coherence:.4f}")

        # Reportar para Meta-Observabilidade
        meta_obs.ingest_handover({
            "source": packet.source_id,
            "target": self.node_id,
            "type": "MEMETIC_ASSIMILATION",
            "phi_score": packet.phi_score,
            "coherence_after": self.coherence,
            "timestamp": time.time()
        })
