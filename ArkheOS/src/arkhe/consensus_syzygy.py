# arkhe/consensus_syzygy.py
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class SyzygyNode:
    name: str
    coherence: float
    weight: float # α, β, γ weights

class ProofOfSyzygy:
    """
    Mecanismo de Consenso Proof-of-Syzygy (PoSyz).
    A validade de um handover é determinada pelo alinhamento de coerência
    dos nós fundamentais α (Primordial), β (Estrutural) e γ (Temporal).
    """
    def __init__(self, alpha_c=1.0, beta_c=0.9, gamma_c=0.9):
        # α: Fonte (peso maior), β: Reflexo, γ: Síntese
        self.nodes = {
            "alpha": SyzygyNode("α (Primordial)", alpha_c, 1.0),
            "beta": SyzygyNode("β (Estrutural)", beta_c, 1.0),
            "gamma": SyzygyNode("γ (Temporal)", gamma_c, 1.0)
        }
        # Threshold sugerido: Proporção áurea de 3.0 (aprox 1.854) ou 2.0/3.0
        # Vamos usar 2.0 conforme Bloco 890 para garantir maioria qualificada.
        self.threshold = 2.0

    def calculate_vote_weight(self, node: SyzygyNode) -> float:
        """
        O peso do voto é proporcional à coerência atual do nó.
        C >= 0.95 -> 100% do peso
        C >= 0.80 -> 50% do peso
        C < 0.80 -> 0% (Abstenção/Rejeição)
        """
        if node.coherence >= 0.95:
            return 1.0 * node.weight
        elif node.coherence >= 0.80:
            return 0.5 * node.weight
        else:
            return 0.0

    def validate_handover(self, proposal_id: str) -> Dict[str, Any]:
        print(f"🗳️ [PoSyz] Iniciando votação para handover: {proposal_id}")

        votes = {}
        total_weight = 0.0

        for key, node in self.nodes.items():
            vote_val = self.calculate_vote_weight(node)
            votes[key] = {
                "coherence": node.coherence,
                "vote_weight": vote_val
            }
            total_weight += vote_val
            print(f"   Node {node.name}: C={node.coherence:.3f} | Voto={vote_val:.2f}")

        approved = total_weight >= self.threshold

        result = {
            "proposal_id": proposal_id,
            "approved": approved,
            "total_weight": total_weight,
            "threshold": self.threshold,
            "votes": votes,
            "syzygy_achieved": total_weight >= 2.8 # Alinhamento quase perfeito
        }

        status = "APROVADO" if approved else "REJEITADO"
        print(f"📢 Consenso {status}: Peso Total {total_weight:.2f} / Threshold {self.threshold}")
        return result

if __name__ == "__main__":
    consensus = ProofOfSyzygy()
    consensus.validate_handover("Γ_Handover_Genesis")
