# ArkheOS Global Entity Registry (Π_1)
# Resolving the "Split-Brain" in Parallel Processing

from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4, UUID
from datetime import datetime
from pydantic import BaseModel, Field
from arkhe.extraction import Provenance, FinancialFact
from enum import Enum

class EntityType(str, Enum):
    FINANCIAL = "financial"
    TECHNICAL_PARAMETER = "technical_parameter"
    LEGAL_CLAUSE = "legal_clause"

class EntityState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    CONFLICTED = "conflicted"

class Entity(BaseModel):
    """A reconciled entity - a stable fact across multiple document chunks."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: EntityType
    value: Any
    unit: Optional[str] = None
    state: EntityState = EntityState.TENTATIVE
    confidence: float = 0.0
    provenance_chain: List[Provenance] = Field(default_factory=list)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    resolution_log: List[str] = Field(default_factory=list)

    def add_provenance(self, prov: Provenance):
        self.provenance_chain.append(prov)
        self.last_seen = datetime.utcnow()
        # Incremental confidence update
        self.confidence = 1.0 - (1.0 - self.confidence) * (1.0 - prov.confidence)

class EntityCandidate(BaseModel):
    """An extraction hypothesis from a single chunk."""
    name: str
    entity_type: EntityType
    value: Any
    unit: Optional[str] = None
    confidence: float
    provenance: Provenance
    chunk_id: str

class GlobalEntityRegistry:
    """Shared work memory for reconciling entities across parallel tasks."""
    def __init__(self, similarity_threshold: float = 0.85):
        self.entities: Dict[UUID, Entity] = {}
        self.name_index: Dict[str, List[UUID]] = {}
        self.threshold = similarity_threshold

    def _canonical_name(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("ψ", "psi")

    def ingest_candidate(self, candidate: EntityCandidate) -> Tuple[Entity, bool]:
        """Reconciles a candidate into the global state."""
        canon = self._canonical_name(candidate.name)

        # Search for existing entities with the same canonical name
        entity_ids = self.name_index.get(canon, [])

        if not entity_ids:
            # Create new entity
            new_entity = Entity(
                name=candidate.name,
                entity_type=candidate.entity_type,
                value=candidate.value,
                unit=candidate.unit,
                confidence=candidate.confidence,
                provenance_chain=[candidate.provenance]
            )
            self.entities[new_entity.id] = new_entity
            self.name_index[canon] = [new_entity.id]
            return new_entity, False

        # Attempt to match with existing (simplified for simulation)
        entity = self.entities[entity_ids[0]]

        # Check for value conflict
        if entity.value != candidate.value:
            entity.state = EntityState.CONFLICTED
            entity.resolution_log.append(f"Conflict: {entity.value} vs {candidate.value}")
            entity.add_provenance(candidate.provenance)
            return entity, True
        else:
            entity.state = EntityState.CONFIRMED
            entity.add_provenance(candidate.provenance)
            return entity, False

    def resolve_manually(self, entity_id: UUID, chosen_value: Any, reason: str):
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity.value = chosen_value
            entity.state = EntityState.CONFIRMED
            entity.resolution_log.append(f"Manual resolution: {reason}")
