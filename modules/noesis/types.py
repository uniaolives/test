# modules/noesis/types.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

# Arkhe Constants
PHI = 0.618033988749895

@dataclass
class CorporateDecision:
    id: str
    content: Any
    criticality: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    affects_cognitive_state: bool = False
    capability_level: str = "BASIC"
    requires_approval: bool = False
    is_distributed: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "content": str(self.content),
            "criticality": self.criticality,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class CorporateIntention:
    id: str
    domain: str
    source: str
    target: str
    content: Any

@dataclass
class EthicalConstraint:
    name: str
    rules: List[str]

    def verify(self, proposal: Any) -> bool:
        return True # Mock verification

class CorporateTreasury:
    def __init__(self, initial_capital: float):
        self.balance = initial_capital

    def allocate(self, amount: float):
        self.balance -= amount

class HumanCouncil:
    def review(self, proposal: Any) -> bool:
        print(f"  [HumanCouncil] Reviewing proposal: {proposal.id}")
        return True

    def alert_violation(self, proposal_id: str, reason: str):
        print(f"  [HumanCouncil] ⚠️ VIOLATION DETECTED in {proposal_id}: {reason}")

class MongoDB:
    def __init__(self, regime: str = 'CRITICAL', C: float = 0.618, F: float = 0.382, h11: int = 491):
        self.regime = regime
        self.data = []

    def record_decision(self, decision: Any):
        self.data.append(decision)

    def insert_one(self, entry: Dict):
        self.data.append(entry)

    def record(self, entry: Dict):
        self.data.append(entry)

class MySQL:
    def __init__(self, regime: str = 'DETERMINISTIC', C: float = 0.9, F: float = 0.1):
        self.regime = regime
        self.data = []

    def record(self, entry: Dict):
        self.data.append(entry)

class Redis:
    def __init__(self, regime: str = 'STOCHASTIC', C: float = 0.3, F: float = 0.7, ttl: int = 3600):
        self.regime = regime
        self.data = {}

    def set(self, key: str, value: Any):
        self.data[key] = value

class CYGeometry:
    def __init__(self, h11: int):
        self.h11 = h11
