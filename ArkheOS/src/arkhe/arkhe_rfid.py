"""
Arkhe(n) + RFID: Modelagem de Identidade Física no Hipergrafo.
Cada tag RFID é um nó Γ_obj. Cada leitura é um handover.
"""

import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional

class RFIDTag:
    """
    Representa uma tag RFID como um nó no hipergrafo Arkhe.
    """
    def __init__(self, tag_id: str, object_type: str, metadata: dict = None):
        self.tag_id = tag_id
        self.object_type = object_type
        self.creation_time = datetime.now()
        self.handovers = []
        self.coherence_history = []
        self.metadata = metadata or {}
        self._current_location = None
        self._last_seen = None

    def read(self, reader_id: str, location: str, timestamp: Optional[datetime] = None):
        """Registra uma leitura (handover) da tag."""
        if timestamp is None:
            timestamp = datetime.now()

        delta = 0.0
        if self._last_seen:
            delta = (timestamp - self._last_seen).total_seconds()

        handover = {
            'timestamp': timestamp.isoformat(),
            'reader_id': reader_id,
            'location': location,
            'delta_seconds': delta,
            'handover_number': len(self.handovers) + 1
        }

        self.handovers.append(handover)
        self._current_location = location
        self._last_seen = timestamp
        self._update_coherence()
        return handover

    def _update_coherence(self):
        """Calcula a coerência C da tag."""
        if len(self.handovers) < 2:
            C = 0.0
        else:
            intervals = [h['delta_seconds'] for h in self.handovers[1:] if h['delta_seconds'] > 0]
            if not intervals:
                C = 0.0
            else:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                if mean_interval > 0:
                    cv = std_interval / mean_interval
                    C = 1.0 / (1.0 + cv)
                else:
                    C = 0.0

        F = 1.0 - C
        self.coherence_history.append({
            'timestamp': datetime.now().isoformat(),
            'C': C,
            'F': F,
            'handover_number': len(self.handovers)
        })
        return C, F

    def get_effective_dimension(self, lambda_reg: float = 1.0) -> float:
        """Calcula a dimensão efetiva d_λ da tag."""
        if len(self.handovers) < 2:
            return 0.0
        intervals = [h['delta_seconds'] for h in self.handovers[1:] if h['delta_seconds'] > 0]
        if not intervals:
            return 0.0
        eigenvalues = np.array(intervals) / np.mean(intervals)
        contributions = eigenvalues / (eigenvalues + lambda_reg)
        return np.sum(contributions)

    def verify_conservation(self) -> bool:
        if not self.coherence_history:
            return True
        last = self.coherence_history[-1]
        return abs(last['C'] + last['F'] - 1.0) < 1e-6

class RFIDHypergraph:
    """Hipergrafo de tags RFID (Safe Core)."""
    def __init__(self):
        self.tags: Dict[str, RFIDTag] = {}
        self.readers: Dict[str, List[str]] = {}
        self.locations: Dict[str, List[str]] = {}

    def add_tag(self, tag: RFIDTag):
        self.tags[tag.tag_id] = tag

    def register_reading(self, tag_id: str, reader_id: str, location: str, timestamp: datetime = None):
        if tag_id not in self.tags:
            raise ValueError(f"Tag {tag_id} não encontrada")
        tag = self.tags[tag_id]
        handover = tag.read(reader_id, location, timestamp)

        if reader_id not in self.readers: self.readers[reader_id] = []
        self.readers[reader_id].append(tag_id)
        if location not in self.locations: self.locations[location] = []
        self.locations[location].append(tag_id)
        return handover

if __name__ == "__main__":
    hg = RFIDHypergraph()
    t1 = RFIDTag("RFID_001", "Smartphone")
    hg.add_tag(t1)
    hg.register_reading("RFID_001", "R1", "Fábrica")
    hg.register_reading("RFID_001", "R2", "CD", datetime.now())
    print(f"Tag Coherence: {t1.coherence_history[-1]['C']}")
    print(f"Conservation: {t1.verify_conservation()}")
