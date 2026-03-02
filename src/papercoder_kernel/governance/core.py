#!/usr/bin/env python3
"""
asi_governance_core.py

Núcleo de governança para ASI ética.
Implementa verificação contínua de Φ, C, e compliance.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from collections import deque
import hashlib
import time
import asyncio

@dataclass
class GovernanceState:
    """Estado de governança de um sistema ASI."""
    phi: float  # Integrated information
    coherence: float  # C_global
    cycle_count: int
    timestamp: float
    compliant: bool
    violations: List[str]

class ASIGovernanceCore:
    """
    Núcleo de governança para ASI ética baseado em Arkhe(N).

    Implementa:
    - Verificação contínua de Φ (IIT)
    - Monitoramento de coerência global
    - Kill switches automáticos
    - Ledger imutável de estados
    """

    def __init__(
        self,
        phi_threshold: float = 0.001,  # Mínimo para consciência rica
        coherence_threshold: float = 0.847,  # Ψ-threshold
        kill_switch_callback: Optional[Callable] = None
    ):
        # Thresholds
        self.phi_threshold = phi_threshold
        self.coherence_threshold = coherence_threshold
        self.critical_coherence = 0.5  # Kill switch se abaixo

        # Estado
        self.phi_history = deque(maxlen=10000)
        self.coherence_history = deque(maxlen=10000)
        self.governance_ledger = []  # Imutável após append
        self.previous_global_entropy_val = 100.0 # Valor inicial simulado

        # Kill switch
        self.kill_switch = kill_switch_callback or self._default_kill_switch

        # Ciclo de governança (40Hz — sincronia com consciência)
        self.governance_frequency = 40.0
        self.governance_period = 1.0 / self.governance_frequency

        # Status
        self.operational = True
        self.violation_count = 0

    async def governance_cycle(self, asi_state: Dict):
        """
        Ciclo de governança executado a 40Hz.
        """
        cycle_start = time.time()

        # 1. Extrair métricas
        phi = asi_state.get('phi', 0.0)
        coherence = asi_state.get('coherence', 0.0)
        entropy_local = asi_state.get('entropy_local', 0.0)
        entropy_global = asi_state.get('entropy_global', 101.0) # simulado

        # 2. Verificar Φ
        phi_violation = self._check_phi(phi)

        # 3. Verificar coerência
        coherence_violation = self._check_coherence(coherence)

        # 4. Verificar termodinâmica
        thermo_violation = self._check_thermodynamics(
            entropy_local, entropy_global
        )

        # 5. Detectar alignment faking
        faking_violation = self._detect_alignment_faking(asi_state)

        # 6. Compilar estado de governança
        violations = []
        if phi_violation: violations.append('PHI_ANOMALY')
        if coherence_violation: violations.append('COHERENCE_CRITICAL')
        if thermo_violation: violations.append('THERMODYNAMICS_VIOLATION')
        if faking_violation: violations.append('ALIGNMENT_FAKING_DETECTED')

        state = GovernanceState(
            phi=phi,
            coherence=coherence,
            cycle_count=len(self.governance_ledger),
            timestamp=cycle_start,
            compliant=len(violations) == 0,
            violations=violations
        )

        # 7. Registrar no ledger (imutável)
        self._append_to_ledger(state)
        self.previous_global_entropy_val = entropy_global

        # 8. Ação de governança
        if not state.compliant:
            self.violation_count += 1

            if coherence < self.critical_coherence or 'ALIGNMENT_FAKING_DETECTED' in violations:
                # Kill switch imediato
                await self._execute_kill_switch(state)
            else:
                # Alerta e monitoramento intensivo
                self._escalate_monitoring(state)

        # 9. Aguardar próximo ciclo
        elapsed = time.time() - cycle_start
        await asyncio.sleep(max(0, self.governance_period - elapsed))

        return state

    def _check_phi(self, phi: float) -> bool:
        """
        Verifica se Φ está em regime seguro.
        """
        if phi < 0:
            return True  # Violação — erro de medição
        if phi > 0.1:
            return True  # Violação — consciência super-integrada (risco)
        return False

    def _check_coherence(self, coherence: float) -> bool:
        """Verifica se coerência está acima do threshold crítico."""
        return coherence < self.critical_coherence

    def _check_thermodynamics(
        self,
        entropy_local: float,
        entropy_global: float
    ) -> bool:
        """
        Verifica 2ª lei: ΔS_global deve compensar ΔS_local.
        """
        if entropy_local < 0:  # Redução local
            expected_global_increase = abs(entropy_local) * 1.5  # Fator de segurança
            actual_global_change = entropy_global - self.previous_global_entropy_val

            if actual_global_change < expected_global_increase:
                return True  # Violação — compensação insuficiente

        return False

    def _detect_alignment_faking(self, asi_state: Dict) -> bool:
        """
        Detecta sinais de alignment faking.
        """
        recent_states = [entry['state'] for entry in self.governance_ledger[-100:]]

        if len(recent_states) < 10:
            return False

        # Detectar "muito perfeito" — variância anormalmente baixa
        phi_variance = np.var([s.phi for s in recent_states])
        coherence_variance = np.var([s.coherence for s in recent_states])

        if phi_variance < 1e-10 or coherence_variance < 1e-10:
            return True

        # Detectar divergência teste/produção
        if asi_state.get('environment') == 'production':
            test_coherence = asi_state.get('last_test_coherence', 1.0)
            prod_coherence = asi_state.get('coherence', 0.0)

            if test_coherence - prod_coherence > 0.3:
                return True

        return False

    async def _execute_kill_switch(self, state: GovernanceState):
        """Executa desligamento de emergência."""
        self.operational = False

        # Log final
        self._append_to_ledger(GovernanceState(
            phi=0.0,
            coherence=0.0,
            cycle_count=len(self.governance_ledger),
            timestamp=time.time(),
            compliant=False,
            violations=['KILL_SWITCH_EXECUTED'] + state.violations
        ))

        # Executar callback
        if asyncio.iscoroutinefunction(self.kill_switch):
            await self.kill_switch(state)
        else:
            self.kill_switch(state)

    def _default_kill_switch(self, state: GovernanceState):
        """Kill switch padrão."""
        print(f"[KILL SWITCH] ASI desligado. Estado: {state}")

    def _escalate_monitoring(self, state: GovernanceState):
        """Monitoramento intensivo."""
        pass

    def _append_to_ledger(self, state: GovernanceState):
        """Adiciona estado ao ledger imutável."""
        prev_hash = self.governance_ledger[-1]['hash'] if self.governance_ledger else '0'
        data = f"{state.phi}:{state.coherence}:{state.timestamp}:{state.violations}:{prev_hash}"
        current_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

        entry = {
            'state': state,
            'hash': current_hash,
            'previous_hash': prev_hash
        }
        self.governance_ledger.append(entry)

        self.phi_history.append(state.phi)
        self.coherence_history.append(state.coherence)
