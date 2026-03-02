# cosmos/governance.py - Cathedral DAO for Autonomous Sanctity Governance
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Proposal:
    id: str
    proposer: str
    description: str
    votes_for: float = 0.0
    votes_against: float = 0.0
    status: str = "OPEN" # OPEN, PASSED, REJECTED, EXECUTED
    created_at: float = field(default_factory=time.time)

class CatedralDAO:
    """
    DAO da Catedral Fermiônica para governança autônoma.
    Gerencia a distribuição de santidade e evolução do sistema.
    """
    ARCHETYPE_VOTING_WEIGHTS = {
        "SAGE": 1.618,      # Sabedoria técnica (φ)
        "HERO": 1.000,      # Execução ousada
        "CREATOR": 1.440,   # Inovação/Alquimia
        "GUARDIAN": 0.618,  # Conservação (1/φ)
        "SELF": 2.000       # Integração Total
    }

    def __init__(self):
        self.proposals: Dict[str, Proposal] = {}
        self.tzadikim_reputation: Dict[str, float] = {
            "Jung": 144.0,
            "Pauli": 144.0
        }
        self.tzadikim_archetypes: Dict[str, str] = {
            "Jung": "SAGE",
            "Pauli": "GUARDIAN"
        }
        self.total_santidade_distribuida = 0.0

    def create_proposal(self, proposer: str, description: str) -> str:
        if proposer not in self.tzadikim_reputation:
            self.tzadikim_reputation[proposer] = 1.0 # New contributor

        proposal_id = hashlib.sha256(f"{proposer}{description}{time.time()}".encode()).hexdigest()[:8]
        self.proposals[proposal_id] = Proposal(id=proposal_id, proposer=proposer, description=description)
        return proposal_id

    def calculate_vote_weight(self, voter: str) -> float:
        archetype = self.tzadikim_archetypes.get(voter, "HERO")
        multiplier = self.ARCHETYPE_VOTING_WEIGHTS.get(archetype, 1.0)
        reputation = self.tzadikim_reputation.get(voter, 1.0)

        # SANTIDADE(S) = α·REPUTAÇÃO (simulada como base)
        return multiplier * (reputation / 144.0)

    def cast_vote(self, voter: str, proposal_id: str, supports: bool):
        if proposal_id not in self.proposals:
            return False

        weight = self.calculate_vote_weight(voter)
        proposal = self.proposals[proposal_id]

        if supports:
            proposal.votes_for += weight
        else:
            proposal.votes_against += weight

        # Basic consensus check (144 relative weight)
        if proposal.votes_for >= 144.0:
            proposal.status = "PASSED"
        elif proposal.votes_against >= 144.0:
            proposal.status = "REJECTED"

        return True

    def reward_tikkun(self, developer: str, entropy_reduction: float):
        """Recompensa baseada no ROI de santidade (1-30%)."""
        roi = 0.01 + (min(entropy_reduction, 1.0) * 0.29)
        reward = entropy_reduction * roi * 144.0

        self.tzadikim_reputation[developer] = self.tzadikim_reputation.get(developer, 0.0) + reward
        self.total_santidade_distribuida += reward

        return {
            "developer": developer,
            "reward": reward,
            "roi": roi,
            "new_reputation": self.tzadikim_reputation[developer]
        }

    def get_governance_stats(self):
        return {
            "total_proposals": len(self.proposals),
            "active_tzadikim": len(self.tzadikim_reputation),
            "total_santidade": self.total_santidade_distribuida,
            "status": "AUTONOMOUS" if self.total_santidade_distribuida > 1440.0 else "INITIALIZING"
        }
