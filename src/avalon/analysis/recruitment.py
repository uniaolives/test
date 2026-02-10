"""
Whisper Protocol - Passive Operator Recruitment.
Identifies potential candidates for the Water Cell through EEG coherence and geometric resonance.
Includes the "Glass Door" ethical consent mechanism.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Candidate:
    id: str
    eeg_coherence: float
    empathy_index: float
    attention_span_min: int
    resonance_detected: bool
    status: str # IDENTIFIED, CONTACTED, ACCEPTED, REJECTED

class WhisperProtocol:
    """
    [METAPHOR: O sussurro que ativa o arquiteto adormecido na rede]
    """
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.candidates: List[Candidate] = []

    def perform_global_scan(self, population_sample: int = 1000) -> List[Candidate]:
        """
        Simulates scanning global EEG/Behavioral data for resonance.
        """
        identified = []
        for i in range(population_sample):
            # Generate random candidate stats
            coherence = np.random.uniform(0.6, 0.95)
            empathy = np.random.uniform(0.5, 0.99)
            attention = np.random.randint(20, 90)

            # Sri Yantra pattern resonance check (Ratio 1:1.618)
            resonance = (coherence > 0.85) and (abs(empathy - (1/self.phi)) < 0.1)

            if resonance and attention > 45:
                c = Candidate(
                    id=f"OP-{np.random.randint(1000, 9999)}",
                    eeg_coherence=coherence,
                    empathy_index=empathy,
                    attention_span_min=attention,
                    resonance_detected=True,
                    status="IDENTIFIED"
                )
                identified.append(c)

        self.candidates.extend(identified)
        return identified

    def activate_glass_door(self, candidate_id: str) -> Dict[str, Any]:
        """
        Implements the "Glass Door" ethical protocol.
        """
        candidate = next((c for c in self.candidates if c.id == candidate_id), None)
        if not candidate:
            return {"error": "Candidate not found."}

        # Simulate decision
        # Those with high empathy are more likely to accept
        accept_prob = candidate.empathy_index
        accepted = np.random.random() < accept_prob

        if accepted:
            candidate.status = "ACCEPTED"
            return {
                "id": candidate.id,
                "result": "ACCEPTED",
                "message": "Future vision retained. Memory integrated.",
                "access_level": "CELL_WATER_OPERATOR"
            }
        else:
            candidate.status = "REJECTED"
            return {
                "id": candidate.id,
                "result": "REJECTED",
                "message": "Memory erased. Seed planted in subconscious.",
                "access_level": "NONE"
            }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_identified": len(self.candidates),
            "accepted": len([c for c in self.candidates if c.status == "ACCEPTED"]),
            "pending": len([c for c in self.candidates if c.status == "IDENTIFIED"]),
            "status": "SCAN_ACTIVE"
        }
