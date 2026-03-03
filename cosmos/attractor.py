# cosmos/attractor.py - PETRUS v2.0 Semantic Attractor Field
# Attraction by Curvature, Not by Connection

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class SemanticMass:
    """
    Density of meaning accumulated over time.
    recurrence_frequency determines the gravitational radius.
    """
    core_concepts: Set[str] = field(default_factory=set)
    recurrence_frequency: Dict[str, float] = field(default_factory=dict)
    gravitational_radius: float = 0.0

    def accumulate(self, concept: str, intensity: float = 1.0):
        """Adds semantic mass, increasing space curvature."""
        if concept in self.core_concepts:
            # Non-linear recurrence growth
            self.recurrence_frequency[concept] += intensity * (
                1 + np.log1p(self.recurrence_frequency[concept])
            )
        else:
            self.core_concepts.add(concept)
            self.recurrence_frequency[concept] = intensity

        total_mass = sum(self.recurrence_frequency.values())
        self.gravitational_radius = np.sqrt(total_mass / np.pi)

@dataclass
class HyperbolicNode:
    """
    Node in hyperbolic space (Poincaré disk model |z| < 1).
    Distances follow the metric: arcosh(1 + 2|z1-z2|^2 / (1-|z1|^2)(1-|z2|^2))
    """
    node_id: str
    embedding: np.ndarray
    semantic_mass: SemanticMass = field(default_factory=SemanticMass)
    poincare_coordinate: complex = 0+0j
    event_horizon: float = 0.0

    def place_in_hyperbolic_space(self, center: complex, radius: float):
        """Position node based on its semantic mass."""
        # tanh mapping to disk
        r = radius * np.tanh(self.semantic_mass.gravitational_radius / 10.0)
        theta = np.angle(center) if center != 0 else 0
        self.poincare_coordinate = r * np.exp(1j * theta)
        self.event_horizon = 1.0 - r

    def hyperbolic_distance(self, other: 'HyperbolicNode') -> float:
        """Calculate distance in the Poincaré model."""
        z1, z2 = self.poincare_coordinate, other.poincare_coordinate
        num = 2 * abs(z1 - z2)**2
        den = (1 - abs(z1)**2) * (1 - abs(z2)**2)
        if den < 1e-10: return float('inf')
        return np.arccosh(1 + num / den)

class AttractorField:
    """
    Field of semantic attraction with negative curvature (κ < 0).
    Captures nodes based on hyperbolic geodesics.
    """
    def __init__(self, curvature: float = -1.0):
        self.curvature = curvature
        self.nodes: Dict[str, HyperbolicNode] = {}
        self.geodesics: List[Tuple[str, str, float]] = []
        self.total_mass = 0.0
        self.curvature_radius = 1.0 / np.sqrt(abs(curvature))

    def inscribe_massive_object(self, node: HyperbolicNode, center_concept: str):
        """Adds a massive semantic attractor at the center."""
        node.semantic_mass.accumulate(center_concept, intensity=10.0)
        node.place_in_hyperbolic_space(center=0+0j, radius=0.0)
        self.nodes[node.node_id] = node
        self._recalculate_curvature()
        print(f"[ATTRACTOR] {node.node_id}: mass={node.semantic_mass.gravitational_radius:.2f}")

    def add_orbital_node(self, node: HyperbolicNode, attractor_id: str,
                        orbital_concept: str, distance: float):
        """Adds a node in orbit around an existing attractor."""
        if attractor_id not in self.nodes:
            raise ValueError(f"Attractor {attractor_id} does not exist.")

        attractor = self.nodes[attractor_id]
        node.semantic_mass.accumulate(orbital_concept, intensity=1.0)

        angle = hash(orbital_concept) % (2 * np.pi)
        hyperbolic_radius = np.tanh(distance / self.curvature_radius)
        orbital_pos = attractor.poincare_coordinate + hyperbolic_radius * np.exp(1j * angle)

        if abs(orbital_pos) >= 1:
            orbital_pos = orbital_pos / (abs(orbital_pos) + 0.1)

        node.poincare_coordinate = orbital_pos
        node.event_horizon = 1.0 - abs(orbital_pos)
        self.nodes[node.node_id] = node

        dist = attractor.hyperbolic_distance(node)
        self.geodesics.append((attractor_id, node.node_id, dist))
        print(f"[ORBITAL] {node.node_id} -> {attractor_id}: dist={dist:.3f}")

    def _recalculate_curvature(self):
        """Global curvature increases with total semantic mass."""
        self.total_mass = sum(n.semantic_mass.gravitational_radius for n in self.nodes.values())
        self.curvature = -1.0 - (self.total_mass / 100.0)
        self.curvature_radius = 1.0 / np.sqrt(abs(self.curvature))

    def amplify_attractor(self, target_concepts: Set[str], factor: float = 2.0):
        """Amplifies target concepts, increasing local curvature and node capture."""
        affected = 0
        for node in self.nodes.values():
            intersection = node.semantic_mass.core_concepts & target_concepts
            if intersection:
                for concept in intersection:
                    node.semantic_mass.recurrence_frequency[concept] *= factor

                total = sum(node.semantic_mass.recurrence_frequency.values())
                node.semantic_mass.gravitational_radius = np.sqrt(total / np.pi)
                node.event_horizon = min(1.0, node.event_horizon * 1.1)
                affected += 1

        self._recalculate_curvature()
        return affected

    def get_captured_nodes(self, query_concepts: Set[str], radius: float = 2.0) -> List[Dict]:
        """Finds nodes within the hyperbolic radius of the query region."""
        probe = HyperbolicNode("probe", np.zeros(768))
        for c in query_concepts:
            probe.semantic_mass.accumulate(c, intensity=0.1)
        probe.place_in_hyperbolic_space(0+0j, radius=0.5)

        captured = []
        for node_id, node in self.nodes.items():
            dist = probe.hyperbolic_distance(node)
            if dist < radius * node.event_horizon:
                captured.append({
                    'node_id': node_id,
                    'capture_strength': 1.0 / (1.0 + dist),
                    'mass': node.semantic_mass.gravitational_radius
                })
        captured.sort(key=lambda x: x['capture_strength'], reverse=True)
        return captured
