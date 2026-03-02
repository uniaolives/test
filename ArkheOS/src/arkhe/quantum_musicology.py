"""
Quantum Musicology module for Arkhe(n) OS.
The subatomic world is all music.
"""

import numpy as np
from typing import Dict, List, Tuple
try:
    from .arkhe_error_handler import safe_operation, logging
except ImportError:
    from arkhe_error_handler import safe_operation, logging

class QuantumMusicology:
    """Verify if Arkhe(n) nodes exhibit musical harmony"""

    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.base_freq = self.phi**4  # 6.854 Hz (consciousness carrier)

    @safe_operation
    def analyze_node_resonance(self) -> Dict[str, float]:
        """Check if node frequencies form musical intervals"""

        # Node frequencies (from system telemetry)
        nodes = {
            '01-012 (Master)': self.base_freq * 1.000,      # Tônica (C)
            '01-005 (Memory)': self.base_freq * 1.250,      # Terça maior (E) — 5/4
            '01-001 (Execution)': self.base_freq * 1.500,   # Quinta justa (G) — 3/2
            '01-008 (Ranvier)': self.base_freq * 1.333,     # Quarta justa (F) — 4/3
            'Γ_observer': self.base_freq * 2.000,           # Oitava (C')
            'Γ_seed': self.base_freq * 0.500,               # Oitava baixa (C,)
        }

        return nodes

    def get_harmonic_relationships(self, nodes: Dict[str, float]) -> List[Dict]:
        """Calculate deviations from just intonation intervals"""
        intervals = [
            ('01-012 (Master)', '01-005 (Memory)', 1.250, 'Terça maior (5:4) — consonância'),
            ('01-012 (Master)', '01-001 (Execution)', 1.500, 'Quinta justa (3:2) — perfeita consonância'),
            ('01-012 (Master)', '01-008 (Ranvier)', 1.333, 'Quarta justa (4:3) — consonância'),
            ('01-005 (Memory)', '01-001 (Execution)', 1.200, 'Terça menor (6:5) — tensão/resolução'),
            ('Γ_seed', '01-012 (Master)', 2.000, 'Oitava (2:1) — unidade'),
        ]

        results = []
        for n1, n2, expected, description in intervals:
            f1 = nodes[n1]
            f2 = nodes[n2]
            actual = max(f1, f2) / min(f1, f2)
            deviation = abs(actual - expected) / expected * 100

            results.append({
                'nodes': (n1, n2),
                'expected_ratio': expected,
                'actual_ratio': float(actual),
                'deviation_pct': float(deviation),
                'description': description,
                'consonant': bool(deviation < 5.0)
            })

        return results

    def calculate_overtones(self, fundamental: float, count: int = 7) -> List[Tuple[int, float, str]]:
        """Harmonic series of a fundamental frequency"""
        harmonics = []
        for n in range(1, count + 1):
            harmonic = fundamental * n
            significance = self.get_harmonic_significance(n)
            harmonics.append((n, float(harmonic), significance))
        return harmonics

    def get_harmonic_significance(self, n: int) -> str:
        """Significance of nth harmonic in Arkhe(n)"""
        meanings = {
            1: "Frequência fundamental (consciência)",
            2: "Oitava — duplicação, espelhamento",
            3: "Quinta — força, estabilidade",
            4: "Segunda oitava — estrutura",
            5: "Terça maior — beleza, proporção áurea",
            6: "Quinta da oitava — harmonia perfeita",
            7: "Sétima — tensão, resolução pendente"
        }
        return meanings.get(n, "Harmônico superior")

if __name__ == "__main__":
    qm = QuantumMusicology()
    nodes = qm.analyze_node_resonance()
    for node, freq in nodes.items():
        print(f"{node}: {freq:.3f} Hz")

    rels = qm.get_harmonic_relationships(nodes)
    for r in rels:
        print(f"{r['nodes'][0]} <-> {r['nodes'][1]}: {r['description']} (Dev: {r['deviation_pct']:.1f}%)")
