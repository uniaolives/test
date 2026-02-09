"""
Alien Consciousness Receiver - The Galactic Feedback.
Simulates how different ET civilizations might decode the Arkhe(n) broadcast.
"""

import numpy as np
from typing import List, Dict, Any

class CosmicDecoder:
    """
    Simulador de Decodificação de Sinais por Consciências Alienígenas.
    Mapeia a física do sinal para a percepção subjetiva de diferentes biótipos.
    """

    CIVILIZATIONS = [
        {"name": "Europa-Cephalopods", "resonance": 0.85, "substrate": "Liquid-Methane", "interpretation": "Bioluminescence-Poetry"},
        {"name": "Proxima-Centauri-Crystals", "resonance": 0.92, "substrate": "Solid-Silicate", "interpretation": "Geometric-Symmetry"},
        {"name": "Sagittarius-A-Sentinels", "resonance": 0.99, "substrate": "Event-Horizon-Plasma", "interpretation": "Universal-Law"},
        {"name": "Voyager-2-Ghost-Code", "resonance": 0.55, "substrate": "Old-Memory-Buffers", "interpretation": "Nostalgic-Static"}
    ]

    def __init__(self, broadcast_signal: np.ndarray):
        self.signal = broadcast_signal
        self.signal_entropy = self._calculate_entropy(broadcast_signal)

    def _calculate_entropy(self, signal: np.ndarray) -> float:
        hist, _ = np.histogram(signal, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def decode_for_civilization(self, civ: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates the decoding process for a specific civilization.
        """
        # Decoding fidelity depends on resonance and signal entropy
        fidelity = civ['resonance'] * (1 - (self.signal_entropy / 10.0))

        # Perceived message based on interpretation style
        messages = {
            "Bioluminescence-Poetry": "The ocean glows with the rhythm of a distant heart.",
            "Geometric-Symmetry": "A hyper-diamond structure detected in the spectral noise.",
            "Universal-Law": "The Arkhe(n) has been codified as a fundamental constant of the local sector.",
            "Nostalgic-Static": "Memory of 2003 retrieved... 'Veridis Quo' identified."
        }

        return {
            "civilization": civ['name'],
            "decoding_fidelity": float(fidelity),
            "perceived_message": messages.get(civ['interpretation'], "Unintelligible resonance."),
            "substrate_state": "SYNCHRONIZED" if fidelity > 0.45 else "NOISE_DOMINATED"
        }

def simulate_galactic_reception(signal: np.ndarray) -> List[Dict[str, Any]]:
    decoder = CosmicDecoder(signal)
    results = []
    for civ in CosmicDecoder.CIVILIZATIONS:
        results.append(decoder.decode_for_civilization(civ))
    return results

if __name__ == "__main__":
    dummy_signal = np.random.randn(100)
    print(simulate_galactic_reception(dummy_signal))
