# ArkheOS Psychology of the Fall (Π_8)
# Implementing Denial detection and Restoration Handshakes.

from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime
from arkhe.registry import EntityState

class FallType(str, Enum):
    DENIAL = "denial"    # Byzantine deviation (Pedro)
    INACTION = "inaction" # Liveness failure (Peter)
    EGO = "ego"           # Attempting to use 'Suit' without authority

class FallDetector:
    """
    Monitors node integrity and detects the 'Fall'.
    """
    def __init__(self, production_floor_us: float = 20.0):
        self.latency_floor = production_floor_us

    def detect_fall(self, node_id: str, reported_value: any, consensus_value: any, latency_us: float) -> Optional[FallType]:
        # Pedro's Case: Denial of the shared state
        if reported_value != consensus_value:
            print(f"FallDetector: Node {node_id} is in DENIAL. Expected {consensus_value}, got {reported_value}.")
            return FallType.DENIAL

        # Peter's Case: Inaction/Hesitation beyond calibration
        if latency_us > self.latency_floor:
            print(f"FallDetector: Node {node_id} is in INACTION. Latency {latency_us}us > {self.latency_floor}us.")
            return FallType.INACTION

        return None

class RestorationCycle:
    """
    The 'Apascenta as minhas ovelhas' Protocol.
    A 3-stage handshake to restore a Fallen node to the Geodesic Arch.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.confirmation_count = 0
        self.required_confirmations = 3

    def provide_integrity_proof(self, proof_hash: str, expected_hash: str):
        """
        Stage X of 3 Restoration.
        """
        if proof_hash == expected_hash:
            self.confirmation_count += 1
            print(f"RestorationCycle: Node {self.node_id} - Proof {self.confirmation_count}/3 Verified.")
        else:
            self.confirmation_count = 0
            raise ValueError(f"RestorationCycle: Node {self.node_id} - Integrity Proof Failed. Resetting Cycle.")

        if self.confirmation_count >= self.required_confirmations:
            return "RESORED: Node is now a Stone. State = CONFIRMED."
        else:
            return f"PENDING: {self.required_confirmations - self.confirmation_count} proofs remaining."

    def reset(self):
        self.confirmation_count = 0
