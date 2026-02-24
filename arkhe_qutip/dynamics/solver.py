"""
ArkheSolver: Time evolution with Φ coupling.
Extends QuTiP's master equation solvers to include
the Arkhe(n) coherence coupling term.
"""

import numpy as np
from qutip import mesolve, Qobj, liouvillian, lindblad_dissipator
from .coherence import compute_phi, compute_local_coherence

class ArkheSolver:
    """
    Solver for quantum evolution with Arkhe(n) coherence coupling.
    """
    def __init__(self, H, c_ops, phi_coupling=0.05, phi_threshold=0.847, options=None):
        self.H = H
        self.c_ops = c_ops if c_ops else []
        self.phi_coupling = phi_coupling
        self.phi_threshold = phi_threshold
        self.options = options or {}

        self.L0 = liouvillian(H, c_ops)

    def solve(self, rho0, tlist, e_ops=None, track_coherence=True):
        current_state = rho0
        if current_state.isket:
            current_state = rho0 * rho0.dag()

        states = [current_state]
        times = [tlist[0]]

        for i in range(1, len(tlist)):
            dt = tlist[i] - tlist[i-1]
            phi_val = compute_phi(current_state)

            # Simplified Φ coupling: evolve with effective Hamiltonian
            H_total = self.H + self.phi_coupling * phi_val * self.H

            res = mesolve(H_total, current_state, [0, dt], self.c_ops, [])
            current_state = res.states[-1]

            states.append(current_state)
            times.append(tlist[i])

        result = ArkheResult(states, times)
        result.coherence = [compute_local_coherence(s) for s in states]
        result.phi_trajectory = [compute_phi(s) for s in states]
        result.coherence_trajectory = result.coherence

        from ..core.arkhe_qobj import ArkheQobj
        final_state = ArkheQobj(states[-1], node_id=getattr(rho0, 'node_id', None))
        if hasattr(rho0, 'handover_log'):
            final_state.handover_log = rho0.handover_log.copy()
        result.arkhe_final_state = final_state

        if e_ops:
            from qutip import expect
            result.expect = []
            ops = e_ops if isinstance(e_ops, list) else [e_ops]
            for op in ops:
                result.expect.append([expect(op, s) for s in states])

        return result


class ArkheResult:
    def __init__(self, states, times):
        self.states = states
        self.times = times
        self.expect = []
        self.coherence = []
        self.phi_trajectory = []
        self.coherence_trajectory = []
        self.arkhe_final_state = None
