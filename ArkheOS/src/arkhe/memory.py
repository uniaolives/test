# ArkheOS Geodesic Memory (Π_4) - Semantic Extension
# The Hipocampo of the Geodesic Arch with Vector Support

from typing import List, Optional, Tuple, Any, Dict
from uuid import UUID, uuid4
from datetime import datetime
import json
import numpy as np
from pydantic import BaseModel
from arkhe.registry import Entity, EntityType, EntityState

class GeodesicTrace(BaseModel):
    """A consolidated fact in the system's eternal memory with semantic vector."""
    trace_id: UUID
    entity_name: str
    entity_type: EntityType
    value: Any
    unit: Optional[str]
    confidence: float
    domain: str = "general"
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    resolution_log: List[str]
    embedding: Optional[List[float]] = None

class GeodesicMemory:
    """
    Manages long-term persistence and semantic retrieval using simulated vector search.
    """
    def __init__(self):
        self.storage: List[GeodesicTrace] = []

    def _generate_embedding(self, text: str) -> List[float]:
        """Simulates embedding generation. Ensures similarity for similar strings."""
        # Simple token-based embedding for testing similarity
        tokens = set(text.lower().split())
        vec = np.zeros(384)
        for token in tokens:
            seed = sum(ord(c) for c in token) % 2**32
            state = np.random.RandomState(seed)
            vec += state.randn(384)

        if np.linalg.norm(vec) > 0:
            vec /= np.linalg.norm(vec)
        return vec.tolist()

    def store_entity(self, entity: Entity, domain: str = "general"):
        if entity.state != EntityState.CONFIRMED:
            return

        existing = next((t for t in self.storage if t.entity_name == entity.name), None)

        if existing:
            existing.last_seen = datetime.utcnow()
            existing.occurrence_count += 1
            existing.confidence = (existing.confidence + entity.confidence) / 2
        else:
            embedding = self._generate_embedding(entity.name)
            trace = GeodesicTrace(
                trace_id=uuid4(),
                entity_name=entity.name,
                entity_type=entity.entity_type,
                value=entity.value,
                unit=entity.unit,
                confidence=entity.confidence,
                domain=domain,
                first_seen=entity.last_seen,
                last_seen=entity.last_seen,
                occurrence_count=1,
                resolution_log=entity.resolution_log,
                embedding=embedding
            )
            self.storage.append(trace)

    def semantic_recall(self, query_text: str, top_k: int = 3) -> List[Tuple[GeodesicTrace, float]]:
        """Recalls past extractions using cosine similarity on embeddings."""
        if not self.storage:
            return []

        query_vec = np.array(self._generate_embedding(query_text))
        results = []

        for trace in self.storage:
            if trace.embedding:
                trace_vec = np.array(trace.embedding)
                denom = (np.linalg.norm(query_vec) * np.linalg.norm(trace_vec))
                similarity = np.dot(query_vec, trace_vec) / denom if denom > 0 else 0
                results.append((trace, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self):
        return {"total_entities": len(self.storage)}
