# noesis-audit/governance/policies/spending_policy.py
from . import Policy

def treasury_guardrail(tx) -> bool:
    """Nenhum agente pode gastar > 1% do tesouro sem aprovação humana."""
    # Simulação de objeto de transação
    limit = tx.get('treasury_total', 0) * 0.01
    if tx.get('amount', 0) > limit:
        return tx.get('approved_by_human', False)
    return True

SPENDING_POLICY = Policy(
    name="Treasury_Spending_Limit",
    description="Spending cap for autonomous agents at 1% of total treasury",
    guardrail=treasury_guardrail,
    violation_action="block",
    level=2
)
