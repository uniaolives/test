# .arkhe/handover/protocol.py
import hashlib
import time
from typing import Optional, Any
from pathlib import Path
import importlib.util

def load_arkhe_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Instância global de meta-observabilidade
current_file = Path(__file__).resolve()
arkhe_root = current_file.parent.parent.parent.parent
utils_path = arkhe_root / "arscontexta" / ".arkhe" / "utils.py"
utils = load_arkhe_module(utils_path, "arkhe.utils")

meta_obs_path = arkhe_root / "arscontexta" / ".arkhe" / "coherence" / "meta_observability.py"
meta_obs_module = utils.load_arkhe_module(meta_obs_path, "arkhe.meta_obs")
meta_obs = meta_obs_module.MetaObservabilityCore()

class MetaHandover:
    """
    Implementação do Protocolo de Handover Universal Arkhe(N).
    Substitui a pilha de rede tradicional por operações atômicas de coerência.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        payload: bytes,
        coherence_in: float,
        coherence_out: float,
        phi_required: float,
        ttl: int = 5
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.payload = payload
        self.coherence_in = coherence_in
        self.coherence_out = coherence_out
        self.phi_required = phi_required
        self.ttl = ttl
        self.timestamp = time.time()
        self.signature = self._generate_signature()

    def _generate_signature(self) -> str:
        data = f"{self.source_id}:{self.target_id}:{self.payload.hex()}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

    def validate_and_execute(self, source_node: Any, target_node: Any, global_phi: float) -> bool:
        """
        Executa o handover atômico em três estágios e notifica a meta-observabilidade.
        """
        if source_node.coherence < self.coherence_in:
            print(f"[HANDOVER REJECTED] Source coherence low: {source_node.coherence:.4f} < {self.coherence_in}")
            return False

        if target_node.coherence < self.coherence_out:
            print(f"[HANDOVER REJECTED] Target coherence low: {target_node.coherence:.4f} < {self.coherence_out}")
            return False

        if global_phi < self.phi_required:
            print(f"[HANDOVER REJECTED] Global Phi insufficient: {global_phi:.4f} < {self.phi_required}")
            return False

        print(f"[HANDOVER EXECUTED] {self.source_id} -> {self.target_id}")
        target_node.receive_handover(self)

        source_node.coherence = min(1.0, source_node.coherence + 0.001)
        target_node.coherence = min(1.0, target_node.coherence + 0.001)

        meta_obs.ingest_handover({
            "source": self.source_id,
            "target": self.target_id,
            "coherence_after": source_node.coherence,
            "timestamp": self.timestamp
        })

        decision = meta_obs.should_metamorphose()
        if decision:
            print(f"[META-OBS] Metamorphosis decided: {decision}")

        return True

class ArkheNode:
    """Simulação de um nó no meta-sistema."""
    def __init__(self, node_id: str, coherence: float = 0.85):
        self.node_id = node_id
        self.coherence = coherence
        self.received_payloads = []

    def receive_handover(self, handover: MetaHandover):
        self.received_payloads.append(handover.payload)
        print(f"[NODE {self.node_id}] Received payload of {len(handover.payload)} bytes")
