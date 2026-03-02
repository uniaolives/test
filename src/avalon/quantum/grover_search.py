# grover_search.py
"""
Algoritmo de Grover adaptado para busca de padrões neurais no Avalon.
Amplificação quântica de estados de alta coerência.
"""
import numpy as np
from typing import Dict, List, Any

class GroverNeuralSearch:
    """
    [METAPHOR: O Cinzel Quântico que encontra o Eu original no mármore informacional]
    """
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.state_size = 2 ** n_qubits
        self.state = np.ones(self.state_size) / np.sqrt(self.state_size)

    def apply_oracle(self, target_states: List[int]):
        """Marca os estados de 'Dharma' com fase negativa"""
        for target in target_states:
            if 0 <= target < self.state_size:
                self.state[target] *= -1

    def apply_diffusion(self):
        """Inversão sobre a média para amplificar amplitude"""
        avg = np.mean(self.state)
        self.state = 2 * avg - self.state

    def search(self, target_states: List[int], iterations: int = None) -> Dict[str, Any]:
        """Executa a busca quântica simulada"""
        if iterations is None:
            # π/4 * sqrt(N/M)
            n = self.state_size
            m = len(target_states)
            iterations = int(np.round(np.pi/4 * np.sqrt(n/m))) if m > 0 else 1

        # Reset state
        self.state = np.ones(self.state_size) / np.sqrt(self.state_size)

        for _ in range(iterations):
            self.apply_oracle(target_states)
            self.apply_diffusion()

        probs = np.abs(self.state) ** 2
        most_likely = int(np.argmax(probs))

        return {
            'iterations': iterations,
            'most_likely_state': most_likely,
            'probability': float(probs[most_likely]),
            'speedup_factor': np.sqrt(self.state_size) / len(target_states) if target_states else 1.0
        }

def demo_grover():
    searcher = GroverNeuralSearch(n_qubits=10) # 1024 estados
    target = [45] # O estado 0x2D (45E) solicitado pelo Arquiteto
    result = searcher.search(target)
    print(f"⚛️ Grover Search for state {target}:")
    print(f"   Found state: {result['most_likely_state']} with prob {result['probability']:.2%}")
    print(f"   Iterations: {result['iterations']}")

if __name__ == "__main__":
    demo_grover()
