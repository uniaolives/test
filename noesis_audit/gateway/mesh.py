# noesis-audit/gateway/mesh.py
"""
Gateway de Agentes (LLM Mesh) para interceptação e aplicação de políticas.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time

class AgentGateway:
    """
    Camada de abstração que intercepta todas as chamadas de agentes para ferramentas e APIs.
    """

    def __init__(self, policy_engine):
        self.policy_engine = policy_engine
        self.allowed_tools = {
            "treasury_query": {"scope": "read_only", "rate_limit": 100, "level": 0},
            "market_research": {"scope": "read_only", "rate_limit": 1000, "level": 0},
            "transaction": {"scope": "write", "rate_limit": 10, "requires_approval": True, "level": 2}
        }
        self.usage_history = {} # agent_id -> {tool -> [timestamps]}

    def _check_rate_limit(self, agent_id: str, tool: str, limit: int) -> bool:
        now = time.time()
        if agent_id not in self.usage_history:
            self.usage_history[agent_id] = {}
        if tool not in self.usage_history[agent_id]:
            self.usage_history[agent_id][tool] = []

        # Janela de 1 hora
        history = [t for t in self.usage_history[agent_id][tool] if now - t < 3600]
        self.usage_history[agent_id][tool] = history

        return len(history) < limit

    def route(self, agent_id: str, agent_level: int, tool: str, params: Dict[str, Any]):
        """Interpreta e roteia a requisição do agente."""
        if tool not in self.allowed_tools:
            return {"error": "Tool not authorized", "status": "blocked"}

        policy = self.allowed_tools[tool]

        # 1. Verifica nível de autonomia
        if agent_level < policy["level"]:
            return {"error": "Insufficient autonomy level", "status": "blocked"}

        # 2. Verifica rate limit
        if not self._check_rate_limit(agent_id, tool, policy["rate_limit"]):
            return {"error": "Rate limit exceeded", "status": "blocked"}

        # 3. Avalia políticas dinâmicas
        evaluation = self.policy_engine.evaluate(params, agent_level)
        if not evaluation["authorized"]:
            return {"error": "Policy violation", "details": evaluation["violations"], "status": "blocked"}

        # 4. Verifica necessidade de aprovação humana
        if policy.get("requires_approval") and not params.get("approved_by_human"):
            return {
                "status": "pending",
                "approval_required": True,
                "message": "This action requires explicit human approval."
            }

        # 5. Executa (simulado)
        self.usage_history[agent_id][tool].append(time.time())
        return {"status": "success", "tool": tool, "timestamp": datetime.now().isoformat()}
