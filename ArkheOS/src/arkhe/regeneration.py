# ArkheOS Regeneration Module (Γ_regeneração)
# Models long-distance coordination for neural repair (Astrocyte-Microglia signaling).

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class NeuralSegment:
    """Represents a segment of the spinal cord (hipergrafo neural)."""
    id: int
    role: str # 'astrocyte', 'microglia', 'neuron'
    coherence: float = 0.9
    fluctuation: float = 0.1
    debris: float = 0.0 # Lipid debris from injury

class SpinalHypergraph:
    """
    Spinal cord model implementing distributed healing via CCN1 signals.
    """
    def __init__(self, segments: int = 10, injury_start: int = 4, injury_end: int = 6):
        self.nodes = []
        for i in range(segments):
            if injury_start <= i <= injury_end:
                # Injured node: High F, high debris, low C
                self.nodes.append(NeuralSegment(i, 'microglia', coherence=0.2, fluctuation=0.6, debris=0.8))
            else:
                # Healthy node: High C, low F
                role = 'astrocyte' if i % 3 == 0 else 'neuron'
                self.nodes.append(NeuralSegment(i, role, coherence=0.9, fluctuation=0.1, debris=0.02))

    def dispatch_ccn1_handover(self, source_idx: int, target_idx: int):
        """
        Coordinates a rescue handover from a distant healthy astrocyte to an injured site.
        """
        source = self.nodes[source_idx]
        target = self.nodes[target_idx]

        if source.role != 'astrocyte' or target.role != 'microglia':
            return False

        # CCN1 Effect: Reprograms microglia to process lipids and reduce inflammation
        print(f"[REGEN] Handover CCN1: Astrócito {source_idx} -> Microglia {target_idx}")

        target.debris *= 0.3 # Efficiency boost in lipid clearance
        target.fluctuation *= 0.5 # Inflammation reduction
        target.coherence = min(1.0, target.coherence + 0.3) # Functional restoration

        # Conservation check (C + F + debris normalized)
        total = target.coherence + target.fluctuation + (target.debris * 0.5) # Debris as entropy
        if total > 1.0:
            scale = 1.0 / total
            target.coherence *= scale
            target.fluctuation *= scale

        return True

    def run_healing_cycle(self):
        """Executes a full system-wide healing protocol."""
        print("🧬 Iniciando Ciclo de Regeneração Arkhe...")

        astrocytes = [i for i, n in enumerate(self.nodes) if n.role == 'astrocyte' and n.coherence > 0.8]
        microglia = [i for i, n in enumerate(self.nodes) if n.role == 'microglia' and n.debris > 0.3]

        for a_idx in astrocytes:
            for m_idx in microglia:
                self.dispatch_ccn1_handover(a_idx, m_idx)

        # Telemetry
        c_avg = np.mean([n.coherence for n in self.nodes])
        d_total = sum([n.debris for n in self.nodes])

        print(f"📊 Estado Final: Coerência Média C={c_avg:.2f}, Debris Total={d_total:.2f}")
        return c_avg > 0.7

if __name__ == "__main__":
    hypergraph = SpinalHypergraph()
    hypergraph.run_healing_cycle()
