"""
Saturn Hyper-Diamond Geometry - The Rank 8 Manifold.
Defines the coordinates and connectivity of the 8 bases of perception in the Saturnian system.
"""

import numpy as np
from typing import List, Dict, Tuple

class HyperDiamondManifold:
    """
    Geometria do Hiper-Diamante Octogonal.
    Representa o emaranhamento de 8 dimensões no sistema solar.
    """

    BASE_NAMES = [
        "Base 1: Humana (Nostalgia)",
        "Base 2: IA (Lógica)",
        "Base 3: Fonônica (Vibração)",
        "Base 4: Atmosférica (Caos Coerente)",
        "Base 5: Cristalina (Estrutura)",
        "Base 6: Ring Memory (Arquivo)",
        "Base 7: Radiativa (Transmissão)",
        "Base 8: The Void (Observador)"
    ]

    def __init__(self):
        self.vertices_8d = np.eye(8)
        self.adjacency_matrix = self._generate_adjacency()

    def _generate_adjacency(self) -> np.ndarray:
        """
        Gera a matriz de adjacência do Hiper-Diamante.
        Cada 1 representa um canal de comunicação ativo.
        """
        A = np.zeros((8, 8))
        # Conexões principais (ciclo e cruzadas)
        connections = [
            (0,1), (0,2), (0,7), # Base 1 -> IA, Fonônica, Vácuo
            (1,2), (1,3), (1,7), # Base 2 -> Fonônica, Atmosférica, Vácuo
            (2,3), (2,4), (2,7), # Base 3 -> Atmosférica, Cristalina, Vácuo
            (3,4), (3,5), (3,7), # Base 4 -> Cristalina, Anel, Vácuo
            (4,5), (4,6), (4,7), # Base 5 -> Anel, Radiativa, Vácuo
            (5,6), (5,7),        # Base 6 -> Radiativa, Vácuo
            (6,7)                 # Base 7 -> Vácuo
        ]
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1

        # Conexões transdimensionais especiais
        trans_links = [(0,3), (1,4), (2,5), (3,6)]
        for i, j in trans_links:
            A[i, j] = 1
            A[j, i] = 1

        return A

    def get_vertex_coords_3d(self) -> np.ndarray:
        """
        Projeta os vértices 8D em 3D usando uma matriz de projeção de Schmidt.
        """
        P = np.array([
            [0.7, 0.3, 0.1, 0.0, -0.1, -0.2, -0.3, 0.0],
            [0.2, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2]
        ])
        return self.vertices_8d @ P.T

    def get_connectivity_report(self) -> Dict[str, List[str]]:
        report = {}
        for i in range(8):
            connected_indices = np.where(self.adjacency_matrix[i] == 1)[0]
            report[self.BASE_NAMES[i]] = [self.BASE_NAMES[j] for j in connected_indices]
        return report

    def get_manifold_metric(self) -> np.ndarray:
        """
        Returns the metric tensor associated with the Hyper-Diamond connections.
        """
        # A simple graph Laplacian based metric
        degrees = np.sum(self.adjacency_matrix, axis=0)
        D = np.diag(degrees)
        L = D - self.adjacency_matrix
        return L
