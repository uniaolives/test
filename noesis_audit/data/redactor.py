# noesis-audit/data/redactor.py
"""
Mascara dados sensíveis em tempo real (Inline Masking).
"""

from typing import List, Dict, Any
from datetime import datetime

class InlineRedactor:
    """
    Mascara dados sensíveis em prompts e respostas com base na função/nível.
    """

    def __init__(self, sensitive_patterns: Dict[str, str]):
        self.patterns = sensitive_patterns

    def redact_content(self, content: str, user_role: str) -> str:
        """
        Redige conteúdo se o papel do usuário não tiver permissão.
        """
        if user_role == "admin":
            return content

        redacted = content
        for label, pattern in self.patterns.items():
            import re
            redacted = re.sub(pattern, f"[{label}_REDACTED]", redacted)
        return redacted

    def create_audit_entry(self, agent_id: str, resource: str, action: str):
        """Gera metadados auditáveis para o acesso."""
        return {
            "agent": agent_id,
            "resource": resource,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "redacted": True
        }
