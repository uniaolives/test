# ArkheOS Failure Memory and Audit (Π_12)
# "The Memory of the Fall - Precedents for Arbitration."

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class Precedent(BaseModel):
    """A resolved conflict recorded as a legal/semantic precedent."""
    entity_name: str
    conflict_context: str
    resolved_value: Any
    reasoning: str
    practitioner_signature: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PrecedentRegistry:
    """
    Jurisprudencia Geodesica.
    Stores past resolutions to prevent repeated failures.
    """
    def __init__(self):
        self.registry: List[Precedent] = []

    def record_precedent(self, entity_name: str, context: str, value: Any, reason: str, signature: str):
        precedent = Precedent(
            entity_name=entity_name,
            conflict_context=context,
            resolved_value=value,
            reasoning=reason,
            practitioner_signature=signature
        )
        self.registry.append(precedent)
        print(f"PrecedentRegistry: New precedent recorded for '{entity_name}'.")

    def lookup_precedents(self, entity_name: str) -> List[Precedent]:
        return [p for p in self.registry if p.entity_name == entity_name]

class SingularityPreventer:
    """
    Blocks recurring patterns of collapse (Ego/Denial).
    Technically implements a 'Deny-list' of failed state transitions.
    """
    def __init__(self, registry: PrecedentRegistry):
        self.registry = registry

    def is_transition_safe(self, entity_name: str, new_value: Any, context: str) -> bool:
        """Checks if the proposed state transition matches a previously rejected pattern."""
        precedents = self.registry.lookup_precedents(entity_name)
        for p in precedents:
            # If the context is similar and the value was previously corrected to something else
            if context in p.conflict_context and new_value != p.resolved_value:
                print(f"SingularityPreventer: Blocking transition for '{entity_name}'. Potential recurring failure.")
                return False
        return True
