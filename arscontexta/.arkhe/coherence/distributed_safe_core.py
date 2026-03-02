# .arkhe/coherence/distributed_safe_core.py
from typing import List, Any, Dict
import hashlib

class SafeCoreFragment:
    """Fragmento do Safe Core residente em cada nó."""
    def __init__(self, node: Any):
        self.node = node

    def calculate_local_hash(self) -> str:
        """Calcula o hash do estado local para verificação de consenso."""
        data = f"{self.node.node_id}:{self.node.coherence:.6f}"
        return hashlib.sha256(data.encode()).hexdigest()

class DistributedSafeCore:
    """
    Sistema de segurança descentralizado.
    Quebra o paradoxo do observador via consenso triádico e emaranhamento.
    """
    def __init__(self, nodes: List[Any]):
        self.fragments = {n.node_id: SafeCoreFragment(n) for n in nodes}
        self.containment_active = False

    def verify_global_integrity(self, quantum_ledger: Any) -> bool:
        """
        Verifica a integridade global via consenso de 2/3 e prova de emaranhamento físico.
        """
        hashes = [f.calculate_local_hash() for f in self.fragments.values()]

        # Simulação de consenso: 100% concordam se tudo estiver coerente
        consensus_reached = len(set(hashes)) <= (len(hashes) // 3 + 1)

        # Prova de emaranhamento: a rede de segurança deve estar fisicamente conectada
        # (Fidelidade média mínima entre fragmentos)
        entanglement_ok = quantum_ledger.get_entanglement_density() > 0.1

        if not consensus_reached or not entanglement_ok:
            self._initiate_containment()
            return False

        return True

    def _initiate_containment(self):
        """Ativa o modo de contenção (congelamento parcial de handovers)."""
        if not self.containment_active:
            print("🚨 [DISTRIBUTED SAFE CORE] Consensus lost or Entanglement decayed! INITIATING CONTAINMENT.")
            self.containment_active = True
            # Em um sistema real, isso suspenderia handovers externos

    def reset(self):
        self.containment_active = False
