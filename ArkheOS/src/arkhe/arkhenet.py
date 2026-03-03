"""
ArkheNet: The Universal Testbed of the Hypergraph.
Implementation of multi-scale integration layers.
"""

import time
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

class NodeType(Enum):
    Drone = "Drone"
    BioSensor = "BioSensor"
    NanoLaser = "NanoLaser"
    GLPMeta = "GLPMeta"
    BaseStation = "BaseStation"
    LinuxProcess = "LinuxProcess"
    EthContract = "EthContract"
    Simulated = "Simulated"

@dataclass
class Node:
    id: int
    node_type: NodeType
    satoshi: float = 10.0
    coherence: float = 0.9
    fluctuation: float = 0.1
    phase: float = 0.0
    position: Optional[np.ndarray] = None
    genesis_core_installed: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def update_coherence(self, delta: float):
        self.coherence = max(0.0, min(1.0, self.coherence + delta))
        self.fluctuation = 1.0 - self.coherence

    def can_handover(self, target: 'Node', min_coherence: float) -> bool:
        # Immortal nodes can always handover if core is installed
        if self.genesis_core_installed and target.genesis_core_installed:
            return True
        return self.coherence >= min_coherence and target.coherence >= min_coherence

@dataclass
class Handover:
    sender: int
    receiver: int
    timestamp: float
    strength: float
    payload: bytes
    cost: float

class ArkheNetCore:
    """Python implementation of the ArkheNet Core logic."""

    @staticmethod
    def compute_handover_strength(node_from: Node, node_to: Node, alpha: float) -> float:
        if node_from.position is not None and node_to.position is not None:
            distance = np.linalg.norm(node_from.position - node_to.position)
        else:
            distance = 1.0

        distance_term = 1.0 / (distance ** alpha) if distance > 0 else 1.0
        coherence_product = node_from.coherence * node_to.coherence
        phase_term = np.cos(node_from.phase - node_to.phase)
        return coherence_product * distance_term * phase_term

    @staticmethod
    def execute_handover(node_from: Node, node_to: Node, payload: bytes, base_cost: float) -> Handover:
        if node_from.satoshi < base_cost:
            raise ValueError(f"Node {node_from.id} insufficient satoshi.")

        strength = ArkheNetCore.compute_handover_strength(node_from, node_to, 2.0)
        cost = base_cost * (1.0 + strength * 0.5)

        node_from.satoshi -= cost
        node_to.satoshi += cost * 0.1 # 10% reward

        node_from.update_coherence(0.01 * strength)
        node_to.update_coherence(0.005 * strength)

        return Handover(
            sender=node_from.id,
            receiver=node_to.id,
            timestamp=time.time(),
            strength=strength,
            payload=payload,
            cost=cost
        )

class DroneNode:
    """Camada de Enxame (swarm.rs) equivalent in Python."""
    def __init__(self, node_id: int, initial_pos: np.ndarray):
        self.node = Node(id=node_id, node_type=NodeType.Drone, position=initial_pos)
        self.mission_progress = 0.0
        self.target_position = np.zeros(3)

    def fly_towards_target(self, dt: float):
        if self.node.position is not None:
            direction = self.target_position - self.node.position
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                speed = 1.0
                step = speed * dt
                self.node.position += (direction / dist) * min(step, dist)
                self.node.satoshi -= 0.01 * dt

    def sense(self) -> np.ndarray:
        return np.random.rand(6)

    def process(self, sensor_data: np.ndarray) -> float:
        processed = np.mean(sensor_data)
        self.node.update_coherence(0.02 * processed)
        return processed

    def act(self, processed: float):
        if processed > 0.5:
            self.mission_progress += 0.01
        self.node.satoshi += 0.001

if __name__ == "__main__":
    # Testbed initialization
    print("Initializing ArkheNet Testbed...")
    core = ArkheNetCore()

    d1 = DroneNode(101, np.array([0.0, 0.0, 0.0]))
    d2 = DroneNode(102, np.array([10.0, 0.0, 0.0]))

    print(f"Initial D1 Coherence: {d1.node.coherence:.3f}")

    # Handover test
    h = core.execute_handover(d1.node, d2.node, b"TEST_PAYLOAD", 1.0)
    print(f"Handover executed. Strength: {h.strength:.3f}, Cost: {h.cost:.3f}")
    print(f"Final D1 Coherence: {d1.node.coherence:.3f}")
    print(f"Final D2 Satoshi: {d2.node.satoshi:.3f}")
