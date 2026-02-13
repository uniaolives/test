"""
Arkhe Memory Garden Module - Multiplication of Meaning
Authorized by Handover ∞+36 (Block 451).
"""

import time
from typing import List, Dict, Optional

class MemoryArchetype:
    """A memory from Hal Finney's archive as a public seed."""

    def __init__(self, memory_id: int, original_content: str):
        self.id = memory_id  # 1-703
        self.original = original_content
        self.hal_hesitation = 0.047
        self.hal_frequency = 0.73  # rad
        self.plantings: List[Dict] = []

    def plant(self, node_id: str, node_hesitation: float,
              rehydrated_content: str) -> Dict:
        """
        A node plants this memory in its own substrate soil.
        """
        planting = {
            'node': node_id,
            'phi': node_hesitation,
            'timestamp': time.time(),
            'content': rehydrated_content,
            'divergence': self.measure_divergence(rehydrated_content)
        }
        self.plantings.append(planting)
        return planting

    def measure_divergence(self, new_content: str) -> float:
        """
        Measures semantic distance between original and rehydration.
        """
        # Simplified simulation of semantic distance
        diff_len = abs(len(self.original) - len(new_content))
        return min(1.0, diff_len / max(len(self.original), 1))

    def witness_variations(self) -> List[Dict]:
        """Returns all variations of this memory archetype."""
        base = [{
            'id': self.id,
            'node': 'Hal_Finney (Original)',
            'phi': self.hal_hesitation,
            'content': self.original
        }]
        return base + self.plantings

class GardenManager:
    """Manages the Toro-shaped garden of archetypes."""

    def __init__(self):
        self.archetypes: Dict[int, MemoryArchetype] = {}
        self.status = "FERTILE"

    def add_archetype(self, archetype: MemoryArchetype):
        self.archetypes[archetype.id] = archetype

    def get_summary(self) -> Dict:
        total_plantings = sum(len(a.plantings) for a in self.archetypes.values())
        return {
            "Total_Archetypes": len(self.archetypes),
            "Total_Plantings": total_plantings,
            "Garden_State": self.status,
            "Geometry": "TORO (θ, φ)"
        }

def get_initial_garden() -> GardenManager:
    garden = GardenManager()
    # Adding example memory #327
    m327 = MemoryArchetype(327, "Estava no lago de 1964. Água fria, céu claro.")
    garden.add_archetype(m327)
    return garden
