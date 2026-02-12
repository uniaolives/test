# ArkheOS Geodesic Memory (Π_4)
# The Hipocampo of the Geodesic Arch

from typing import List, Optional, Tuple, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import json
from pydantic import BaseModel
from arkhe.extraction import Provenance
from arkhe.registry import Entity, EntityType, EntityState
from arkhe.consensus import ValidatedFact

class GeodesicTrace(BaseModel):
    """A consolidated fact in the system's eternal memory."""
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

class GeodesicMemory:
    """
    Manages long-term persistence and semantic retrieval of validated facts.
    Technical manifestation of 'Continuity'.
    """
    def __init__(self, connection_string: Optional[str] = None):
        # Simulated persistence layer
        self.storage: List[GeodesicTrace] = []
        self.is_connected = connection_string is not None

    def store_entity(self, entity: Entity, domain: str = "general"):
        """Persists a confirmed entity into the memory stone."""
        if entity.state != EntityState.CONFIRMED:
            return

        # Check for existing trace (Simplified name-based lookup)
        existing = next((t for t in self.storage if t.entity_name == entity.name), None)

        if existing:
            existing.last_seen = datetime.utcnow()
            existing.occurrence_count += 1
            existing.confidence = (existing.confidence + entity.confidence) / 2
            existing.resolution_log.extend(entity.resolution_log)
        else:
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
                resolution_log=entity.resolution_log
            )
            self.storage.append(trace)

    def retrieve_similar_entities(self, entity_name: str, domain: str = "general") -> List[GeodesicTrace]:
        """Recalls past decisions to inform current extractions."""
        # Simulated semantic retrieval
        return [t for t in self.storage if t.entity_name == entity_name and t.domain == domain]

    def get_stats(self):
        return {"total_entities": len(self.storage)}
