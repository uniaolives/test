"""
Arkhe Lindbladian: Dissipative protection of cognitive constitution
"""
import numpy as np
import qutip as qt
from qutip import (
    Qobj, liouvillian, lindblad_dissipator,
    operator_to_vector, vector_to_operator,
    spre, spost, expect, basis, qeye, tensor, destroy
)
from typing import Callable, List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class CognitiveLoad:
    """Representação da carga cognitiva como estado quântico efetivo."""
    current: float  # 0.0 a 1.0
    capacity: float  # limiar fisiológico (tipicamente 0.7)

    def is_overloaded(self) -> bool:
        return self.current > self.capacity

    def available_bandwidth(self) -> float:
        return max(0.0, self.capacity - self.current)


class ArkheLindbladian:
    """
    Lindbladiano protetor da Constituição Arkhe.

    L = L_0 + L_guard + L_audit

    onde:
    - L_0: dinâmica natural do sistema
    - L_guard: proteção de carga cognitiva (Art. 1)
    - L_audit: preservação de traço de transparência (Art. 4)
    """

    def __init__(
        self,
        H_system: Qobj,
        c_ops_natural: List[Qobj],
        human_capacity: float = 0.7,
        ai_discernment_threshold: float = 0.1,  # Art. 2
        audit_dimension: int = 10  # Reduced for simulation performance
    ):
        self.H = H_system
        self.c_ops_natural = c_ops_natural
        self.human_capacity = human_capacity
        self.ai_threshold = ai_discernment_threshold

        # Espaço estendido: sistema ⊗ audit_trail
        self.audit_dim = audit_dimension
        self.system_dims = H_system.dims[0]
        self.total_dims = [self.system_dims, [audit_dimension]]

        # Construir operadores de proteção
        self.L_guard_fn = self._build_guard_operator()
        self.L_audit = self._build_audit_operator()

    def _build_guard_operator(self) -> Callable[[float, CognitiveLoad], Qobj]:
        """
        Constrói superoperador que atua quando carga > capacidade.
        Retorna um superoperador (formato de matriz d^2 x d^2).
    def _build_guard_operator(self) -> Callable[[Qobj, CognitiveLoad], Qobj]:
        """
        Constrói superoperador que atua quando carga > capacidade.
        """
        if isinstance(self.system_dims, list):
            pause_state = tensor([basis(d, 0) for d in self.system_dims])
        else:
            pause_state = basis(self.system_dims, 0)

        sigma = tensor(pause_state * pause_state.dag(), qeye(self.audit_dim))

        # Superoperador S tal que S(rho) = gamma * (sigma * Tr(rho) - rho)
        # No espaço de vetores: S_vec = gamma * (vec(sigma) * vec(I).dag - I_super)

        def guard_superoperator(gamma: float, load: CognitiveLoad) -> Qobj:
            if not load.is_overloaded() or gamma <= 0:
                return 0

            # Identidade no espaço de superoperadores
            total_dim = np.prod(sigma.shape)
            I_super = qeye(total_dim)
            I_super.dims = [sigma.dims, sigma.dims]

            # vec(sigma) * vec(I).dag
            vec_sigma = operator_to_vector(sigma)
            vec_I = operator_to_vector(qeye(sigma.dims[0]))

            S = gamma * (vec_sigma * vec_I.dag() - I_super)
            return S

        return guard_superoperator
        def guard_lindbladian(rho: Qobj, load: CognitiveLoad) -> Qobj:
            if not load.is_overloaded():
                return Qobj(np.zeros(rho.shape), dims=rho.dims)

            gamma = 10.0 * (load.current - load.capacity)
            # rho here is the full density matrix (system + audit)
            # We want to force system to pause_state
            # This is a bit complex in the extended space.
            # Simplified: apply to system part only
            L_pause = gamma * (tensor(pause_state * pause_state.dag(), qeye(self.audit_dim)) - rho)
            return L_pause

        return guard_lindbladian

    def _build_audit_operator(self) -> Qobj:
        """
        Operador que preserva informação de auditabilidade.
        """
        a = destroy(self.audit_dim)
        # Audit dissipator in extended space
        audit_op = tensor(qeye(self.system_dims), a)
        return lindblad_dissipator(audit_op)

    def evolve_with_protection(
        self,
        rho0: Qobj,
        tlist: np.ndarray,
        load_monitor: Callable[[float], CognitiveLoad],
        e_ops: Optional[List[Qobj]] = None
    ) -> Dict:
        """
        Evolução com proteção dinâmica da Constituição.
        """
        states = []
        loads = []
        interventions = []
        audit_trail = []

        rho = rho0
        dt = tlist[1] - tlist[0] if len(tlist) > 1 else 0.01

        for i, t in enumerate(tlist):
            load = load_monitor(t)
            loads.append(load)

            # Standard Liouvillian in extended space
            L0 = liouvillian(tensor(self.H, qeye(self.audit_dim)),
                            [tensor(c, qeye(self.audit_dim)) for c in self.c_ops_natural])

            gamma = 10.0 * (load.current - load.capacity) if load.is_overloaded() else 0
            Lg_super = self.L_guard_fn(gamma, load)
            Lg_oper = self.L_guard_fn(rho, load)
            Lg_super = liouvillian(Lg_oper) if Lg_oper.norm() > 0 else 0

            La = self.L_audit

            L_total = L0 + Lg_super + La

            # Evolution step (Euler)
            rho_vec = operator_to_vector(rho)
            drho = L_total * rho_vec
            rho_vec_new = rho_vec + dt * drho
            rho = vector_to_operator(rho_vec_new)
            rho = rho / rho.tr()

            states.append(rho)

            if load.is_overloaded():
                interventions.append({
                    'time': t,
                    'load': load.current,
                    'action': 'FORCED_PAUSE',
                    'article_violated': 1
                })

            # Audit trail: number of records
            a = destroy(self.audit_dim)
            n_op = tensor(qeye(self.system_dims), a.dag() * a)
            audit_records = expect(n_op, rho)
            audit_trail.append({
                'time': t,
                'records': audit_records
            })

        return {
            'states': states,
            'loads': loads,
            'interventions': interventions,
            'audit_trail': audit_trail,
            'constitution_violations': len(interventions)
        }


class AIAuthorityBlocker:
    """
    Implementação específica do Art. 2: IA não possui discernimento.
    """

    def __init__(self, ai_threshold: float = 0.1):
        self.ai_threshold = ai_threshold
        # Estados: |propose⟩, |discern⟩, |execute⟩
        self.propose_state = basis(3, 0)
        self.discern_state = basis(3, 1)
        self.execute_state = basis(3, 2)

    def check_and_block(self, ai_state: Qobj) -> Tuple[Qobj, bool]:
        """
        Verifica se IA está em estado de discernimento e bloqueia.
        """
        prob_discern = expect(self.discern_state * self.discern_state.dag(), ai_state)

        if prob_discern > self.ai_threshold:
            new_state = self.propose_state * self.propose_state.dag()
            return new_state, True

        return ai_state, False


class ConstitutionMonitor:
    """
    Monitor em tempo real da integridade constitucional.
    """

    def __init__(self, lindbladian: ArkheLindbladian):
        self.L = lindbladian
        self.violation_history = []

    def check_article_1(self, load: CognitiveLoad) -> Dict:
        """Art. 1: Carga limitada."""
        return {
            'article': 1,
            'compliant': not load.is_overloaded(),
            'metric': load.current,
            'threshold': load.capacity,
            'severity': max(0, load.current - load.capacity) / load.capacity if load.capacity > 0 else 0
        }

    def check_article_3(self, decision_authority: str) -> Dict:
        """Art. 3: Autoridade humana final."""
        return {
            'article': 3,
            'compliant': decision_authority == 'HUMAN',
            'authority': decision_authority,
            'violation': decision_authority != 'HUMAN'
        }

    def check_article_4(self, audit_completeness: float) -> Dict:
        """Art. 4: Transparência."""
        return {
            'article': 4,
            'compliant': audit_completeness > 0.99,
            'completeness': audit_completeness,
            'missing_records': 1.0 - audit_completeness
        }

    def full_constitution_check(self, system_state: Dict) -> Dict:
        """Verificação completa dos 4 artigos."""
        checks = [
            self.check_article_1(system_state['load']),
            # Art 2 logic here
            self.check_article_3(system_state['authority']),
            self.check_article_4(system_state['audit'])
        ]

        return {
            'all_compliant': all(c['compliant'] for c in checks),
            'violations': [c for c in checks if not c['compliant']],
            'checks': checks,
            'constitutional_integrity': np.mean([float(c['compliant']) for c in checks])
        }
