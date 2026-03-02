# noesis-audit/governance/policies/__init__.py
"""
Framework "Policy as Code" para governança ética e operacional.
"""

from dataclasses import dataclass
from typing import Callable, Any, Optional
from datetime import datetime

@dataclass
class Policy:
    name: str
    description: str
    guardrail: Callable[[Any], bool]
    violation_action: str  # "block", "alert", "quarantine"
    level: int  # autonomia mínima para aplicar

class PolicyEngine:
    def __init__(self):
        self.policies = []

    def add_policy(self, policy: Policy):
        self.policies.append(policy)

    def evaluate(self, action_context: Any, agent_level: int) -> dict:
        results = {"authorized": True, "violations": []}
        for policy in self.policies:
            if agent_level >= policy.level:
                if not policy.guardrail(action_context):
                    results["authorized"] = False if policy.violation_action == "block" else results["authorized"]
                    results["violations"].append({
                        "policy": policy.name,
                        "action": policy.violation_action
                    })
        return results
