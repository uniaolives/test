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

# Carregar utilitários e ledger quântico
utils_path = arkhe_root / "arscontexta" / ".arkhe" / "utils.py"
spec = importlib.util.spec_from_file_location("arkhe.utils", str(utils_path))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

q_ledger_path = arkhe_root / "arscontexta" / ".arkhe" / "ledger" / "quantum_ledger.py"
q_ledger_module = utils.load_arkhe_module(q_ledger_path, "arkhe.q_ledger")
quantum_ledger = q_ledger_module.QuantumLedger()

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
        self.mass = max(0.0, phi_score - 1.0) # Massa efetiva via Higgs mecânico
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
        from collections import deque
        self.packet_buffer = deque(maxlen=10) # Buffer para inibição lateral
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
        Propagação Epidêmica (Gossip) com alcance limitado pela massa (Higgs).
        """
        fanout = 3
        if not self.peers:
            return

        # Modos sem massa (phi <= 1) propagam globalmente (small-world)
        # Modos com massa têm alcance limitado: radius ~ 1/mass
        if packet.mass > 0:
            radius = 1.0 / packet.mass
            # Simulação simplificada de raio de vizinhança na rede Gossip
            # (Em uma rede real, isso usaria coordenadas de rede ou hops)
            targets = [p for p in self.peers if random.random() < radius]
            if len(targets) > fanout:
                targets = random.sample(targets, fanout)
        else:
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

        # Filtro Bayesiano (Inibição Lateral)
        if not self._bayesian_coherence_check(packet):
            return 0.0

        return float(np.clip(resonance, 0, 1))

    def _bayesian_coherence_check(self, new_packet: MemeticPacket) -> bool:
        """
        Inibição lateral como inferência Bayesiana com prior competitiva.
        """
        # Prior: compatibilidade com o estado atual do nó
        dot = np.dot(new_packet.context_vector, self.state_vector)
        likelihood = np.exp(dot)

        # Evidence: normalização sobre pacotes recentes no buffer
        if not self.packet_buffer:
            self.packet_buffer.append(new_packet)
            return True

        evidence = sum([np.exp(np.dot(p.context_vector, self.state_vector))
                       for p in self.packet_buffer])

        # Posterior (simplificado)
        posterior_prob = likelihood / (evidence + 1e-9)

        # Aceitar se for suficientemente 'explanatório' (threshold 0.3)
        accepted = posterior_prob > 0.3
        if accepted:
            self.packet_buffer.append(new_packet)
        else:
            print(f"    [BAYES] Packet {new_packet.id[:8]} rejected by node {self.node_id}. Prob: {posterior_prob:.4f}")
        return accepted

    def gross_pitaevskii_step(self, dt: float, g: float = 0.1):
        """
        Evolução do vetor de estado via Equação de Gross-Pitaevskii Informacional.
        Simula a condensação de conhecimento em um estado fundamental coerente.
        """
        if not self.peers:
            return

        # Laplaciano discreto (difusão de coerência entre vizinhos)
        neighbor_avg = np.mean([p.state_vector for p in self.peers], axis=0)
        laplacian = neighbor_avg - self.state_vector

        # Potencial químico efetivo (baseado na coerência local)
        mu = self.coherence

        # Termo não-linear (auto-interação/metacognição)
        # Usamos uma aproximação clássica para a complexidade quântica
        nonlinear = self.state_vector * np.linalg.norm(self.state_vector)**2

        # Evolução temporal (Parte real da fase de Berry informacional)
        d_state = (laplacian + mu * self.state_vector + g * nonlinear) * dt
        self.state_vector += d_state
        self.state_vector /= np.linalg.norm(self.state_vector) # Normalização (Preservação de Probabilidade)

    def _assimilate(self, packet: MemeticPacket):
        """
        Meta-learning: Assimilação do 'Delta de Sabedoria'.
        """
        print(f"✨ [NODE {self.node_id}] ASSIMILATED Insight from {packet.source_id}!")

        self.knowledge['external_wisdom'] = packet.content

        # Evolução do vetor de estado (Gross-Pitaevskii explícito)
        # O vetor do nó se move na direção do insight (condensação)
        lr = 0.1
        self.state_vector += lr * (packet.context_vector - self.state_vector)
        self.gross_pitaevskii_step(dt=0.1)

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

        # Registrar emaranhamento no Quantum Ledger
        quantum_ledger.record_handover(packet.source_id, self.node_id, packet.phi_score)
