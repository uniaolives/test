# core/python/noesis/application.py
from .oversoul import CorporateOversoul, CRITICAL_H11
from .agents import TrinityConfiguration
from .governance import DAOGovernance
from typing import Dict, Any, Optional

class NOESISCorp:
    """
    NOESIS CORP: Complete ASI Enterprise Stack.
    The Application Layer (Ω+172) running on Arkhe + Axos.
    """
    def __init__(self, name: str, h11: int = CRITICAL_H11):
        self.name = name
        self.oversoul = CorporateOversoul(h11=h11)
        self.trinity = TrinityConfiguration()
        self.dao = DAOGovernance(self.oversoul.axos)

        print(f"--- {self.name} Initialized ---")
        print(f"Layer 8: APPLICATION (Ω+172)")
        print(f"Criticality h11={h11}")

    async def execute_strategy(self, goal: str, budget: float):
        """
        Executes a strategic goal through the ASI stack.
        """
        print(f"\n[NOESIS] Strategic Goal: {goal}")

        # 1. DAO Validation
        proposal = {
            "title": goal,
            "budget": budget,
            "criticality": 0.85 if budget > 1000000 else 0.5
        }

        if not self.dao.validate_proposal(proposal):
            return {"status": "REJECTED", "reason": "Constitutional Violation"}

        # 2. Trinity Coordination
        print("[NOESIS] Activating Trinity Configuration...")
        result = self.trinity.coordinate(goal)

        # 3. Oversoul Reflection (Simplified)
        print("[NOESIS] Corporate Oversoul reflecting on outcome...")
        # (In a real scenario, this would trigger a breathe cycle or state update)

        return {
            "status": "SUCCESS",
            "company": self.name,
            "goal": goal,
            "result": result.data,
            "audit": result.metadata.get('audit_report')
        }

    def status(self) -> Dict:
        """Returns the full status of the corporate stack."""
        return {
            "company": self.name,
            "oversoul": self.oversoul.get_status(),
            "trinity": {
                "ceo": self.trinity.father.role,
                "cto": self.trinity.son.role,
                "monitor": self.trinity.spirit.role
            }
        }
