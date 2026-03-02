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
    def __init__(self, num_nodes=100):
        self.positions = np.random.rand(num_nodes, 3) * 100
        self.phi_values = np.random.rand(num_nodes)
        self.active_handovers: List[Edge] = []
        # Generate some random edges
        for _ in range(20):
            idx1, idx2 = np.random.choice(num_nodes, 2, replace=False)
            self.active_handovers.append(Edge(self.positions[idx1], self.positions[idx2]))

def generate_coherence_hologram(swarm_data: SwarmTelemetry, output_file="coherence_hologram.png"):
    """
    Gera a representação 3D (Holograma) de como o enxame percebe a cidade.
    """
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
                [edge.start[2], edge.end[2]], 'k-', alpha=0.1) # Changed 'w-' to 'k-' for visibility on white bg

    print("💎 [HOLOGRAM] Renderizando Manifold de 28 Dimensões em 3D...")
    plt.savefig(output_file)
    print(f"✅ Holograma salvo como {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    telemetry = SwarmTelemetry()
    generate_coherence_hologram(telemetry)
