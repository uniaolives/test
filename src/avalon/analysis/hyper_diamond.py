"""
Hyper-Diamond Manifold - The Rank 8 connectivity architecture.
Defines the adjacency and 3D projection of the 8 bases of perception.
"""

import numpy as np
from typing import List, Dict, Tuple

class HyperDiamondManifold:
    """
    Geometria do Hiper-Diamante Octogonal.
    Representa o emaranhamento de 8 dimensões no sistema solar.
    """

    BASE_NAMES = [
        "Humana", "IA", "Fonônica", "Atmosférica",
        "Cristalina", "Memória-Anel", "Radiativa", "Vácuo"
    ]

    def __init__(self):
        # Coordinates in R^8
        self.vertices_8d = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.adjacency_matrix = self._generate_adjacency()

    def _generate_adjacency(self) -> np.ndarray:
        """
        Adjacency matrix for the Hyper-Diamond.
        """
        A = np.array([
            [0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0]
        ])
        return A

    def get_vertex_coords_3d(self) -> np.ndarray:
        """
        Projects 8D vertices to 3D using the Schmidt transformation matrix.
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

    def get_edges(self) -> List[Tuple[int, int]]:
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if self.adjacency_matrix[i, j] == 1:
                    edges.append((i, j))
        return edges

    def get_manifold_summary(self) -> Dict:
        return {
            "rank": 8,
            "topology": "HYPERDIAMOND",
            "active_connections": int(np.sum(self.adjacency_matrix) // 2),
            "description": "8 bases of perceptions in simultaneous entanglement"
        }
