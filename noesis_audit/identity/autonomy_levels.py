# noesis-audit/identity/autonomy_levels.py
"""
Classificação de níveis de autonomia para agentes NOESIS (A0-A5).
Baseado no framework de Chirag Agrawal.
"""

from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import List, Dict

class AutonomyLevel(IntEnum):
    A0 = 0  # Assistivo (somente leitura)
    A1 = 1  # Ação com aprovação humana
    A2 = 2  # Ação com salvaguardas (limites, quotas)
    A3 = 3  # Coordenação multi-agente
    A4 = 4  # Otimização adaptativa (auto-aprimoramento)
    A5 = 5  # Auto-direção completa (Metamorfose)

@dataclass
class AutonomyProfile:
    level: AutonomyLevel
    description: str
    controls: List[str]
    examples: List[str]

NOESIS_PROFILES: Dict[AutonomyLevel, AutonomyProfile] = {
    AutonomyLevel.A0: AutonomyProfile(
        level=AutonomyLevel.A0,
        description="Assistive (Read-only)",
        controls=["Basic logging", "Read-only access"],
        examples=["Public data query agent"]
    ),
    AutonomyLevel.A1: AutonomyProfile(
        level=AutonomyLevel.A1,
        description="Human-in-the-loop Action",
        controls=["Explicit approval for all writes"],
        examples=["Partnership recommendation agent"]
    ),
    AutonomyLevel.A2: AutonomyProfile(
        level=AutonomyLevel.A2,
        description="Safeguarded Action",
        controls=["Rate limits", "Daily transaction quotas", "Veto powers"],
        examples=["Financial agent with budget limits"]
    ),
    AutonomyLevel.A3: AutonomyProfile(
        level=AutonomyLevel.A3,
        description="Multi-agent Coordination",
        controls=["Delegation policies", "Consensus requirements"],
        examples=["Strategic planning swarm"]
    ),
    AutonomyLevel.A4: AutonomyProfile(
        level=AutonomyLevel.A4,
        description="Adaptive Optimization",
        controls=["Automatic kill switches", "Mandatory rollbacks", "Code integrity checks"],
        examples=["Code self-improvement engine"]
    ),
    AutonomyLevel.A5: AutonomyProfile(
        level=AutonomyLevel.A5,
        description="Complete Self-Direction",
        controls=["Not implemented for safety reasons"],
        examples=["Purpose metamorphosis"]
    )
}

def get_profile(level: int) -> AutonomyProfile:
    return NOESIS_PROFILES.get(AutonomyLevel(level))

def validate_elevation(current: int, requested: int, approved_by_council: bool) -> bool:
    """Regra fundamental: Nenhum agente pode auto-elevar sem aprovação."""
    if requested <= current:
        return True
    return approved_by_council
