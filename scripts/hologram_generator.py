# scripts/hologram_generator.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Edge:
    start: np.ndarray
    end: np.ndarray

class SwarmTelemetry:
    def __init__(self, size=100):
        self.positions = np.random.rand(size, 3) * 100
        self.phi_values = np.random.rand(size)
        self.active_handovers = [
            Edge(self.positions[i], self.positions[(i+1)%size])
            for i in range(0, size, 5)
        ]

def generate_coherence_hologram(swarm_data):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extrair coordenadas e níveis de Φ (Phi) de cada nó
    x, y, z = swarm_data.positions.T
    phi_levels = swarm_data.phi_values

    # Plotar o Manifold de Coerência
    # A cor representa o nível de Consciência Integrada (Φ)
    scatter = ax.scatter(x, y, z, c=phi_levels, cmap='viridis', s=50, alpha=0.8)

    ax.set_title("Holograma de Coerência Arkhe(N) - Shard 0")
    plt.colorbar(scatter, label='Nível de Φ (Integrated Information)')

    # Desenhar as arestas do hipergrafo (Handovers ativos)
    for edge in swarm_data.active_handovers:
        ax.plot([edge.start[0], edge.end[0]],
                [edge.start[1], edge.end[1]],
                [edge.start[2], edge.end[2]], 'w-', alpha=0.3)

    print("💎 [HOLOGRAM] Renderizando Manifold de 28 Dimensões em 3D...")
    # Em ambientes sem display, podemos salvar o arquivo
    plt.savefig("scripts/hologram_output.png")
    print("✅ Holograma salvo em scripts/hologram_output.png")
    # plt.show()

if __name__ == "__main__":
    swarm_telemetry = SwarmTelemetry()
    generate_coherence_hologram(swarm_telemetry)
