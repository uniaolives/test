import json
from typing import List, Dict

# Simulated ontological memory storage
_ontology = []

def find_similar_concepts(embedding: List[float], top_k: int = 3) -> List[Dict]:
    """
    Simulates a vector search in the ontological memory.
    In production, this uses pgvector.
    """
    # Returning mock data for demonstration
    return [
        {"name": "geodésica", "similarity": 0.95},
        {"name": "cicloide", "similarity": 0.88},
        {"name": "syzygy", "similarity": 0.94}
    ]

def load_ontology(concepts: List[Dict]):
    global _ontology
    _ontology.extend(concepts)

def get_ontology_size() -> int:
    return len(_ontology)
