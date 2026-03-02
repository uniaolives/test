# noesis-audit/data/data_lineage.py
"""
Rastreia a linhagem e o ciclo de vida dos dados na NOESIS.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class DataEvent:
    source: str
    destination: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

class DataLineageTracker:
    def __init__(self):
        self.history: List[DataEvent] = []

    def record_movement(self, src: str, dest: str, tags: List[str]):
        event = DataEvent(source=src, destination=dest, tags=tags)
        self.history.append(event)

    def get_lineage(self, resource_name: str) -> List[DataEvent]:
        return [e for e in self.history if e.destination == resource_name or e.source == resource_name]
