# core/python/axos/orchestration.py
import time
from typing import List, Dict, Any
from .base import Agent, Payload, Handover, HandoverResult, Human, Content, UserResult, InteractionGuard, SystemCall, SystemResult
from .integrity import AxosIntegrityGates
from .deterministic import AxosDeterministicExecution

class AxosAgentOrchestration(AxosIntegrityGates, AxosDeterministicExecution):
    """
    Axos supports three types of agent interactions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_registry = {}
        self.handover_log = []

    def agent_to_agent_handover(self, source: Agent, target: Agent, payload: Payload) -> HandoverResult:
        """Agent-to-Agent communication."""
        if source.id not in self.agent_registry or target.id not in self.agent_registry:
            raise Exception("Agent not registered")

        handover = Handover(source=source, target=target, payload=payload, protocol='YANG_BAXTER')

        if not handover.verify_yang_baxter():
            raise Exception("Yang-Baxter violation")

        result = handover.execute()

        self.handover_log.append({
            'type': 'AGENT_TO_AGENT',
            'source': source.id,
            'target': target.id,
            'payload_hash': hash(str(payload)),
            'timestamp': time.time(),
            'result': result.status
        })

        return result

    def agent_to_user_handover(self, agent: Agent, user: Human, content: Content) -> UserResult:
        """Agent-to-User communication."""
        guard = InteractionGuard(user, agent)

        if not guard.can_process(volume=content.token_count, entropy=content.complexity):
            return UserResult(status='BLOCKED', reason='Cognitive overload protection (ISC > 0.7)')

        output = guard.propose_interaction(content)

        if output is None:
            return UserResult(status='BLOCKED', reason='Human protection (Constitution Art. 5)')

        approval = user.review(output)
        guard.record_review(output, approval)

        return UserResult(status='SUCCESS' if approval else 'REJECTED', output=output if approval else None)

    def agent_to_system_handover(self, agent: Agent, syscall: SystemCall) -> SystemResult:
        """Agent-to-System communication."""
        if not self.integrity_gate(syscall):
            return SystemResult(status='BLOCKED', reason='Integrity gate failure (fail-closed)')

        result = self.deterministic_execute(syscall)

        return SystemResult(status='SUCCESS', result=result)
