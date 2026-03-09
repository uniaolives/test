"""
ArkheSolver: Time evolution with Φ coupling.
Extends QuTiP's master equation solvers to include
the Arkhe(n) coherence coupling term.
"""

import numpy as np
from qutip import mesolve, Qobj, liouvillian
from .coherence import compute_phi, compute_local_coherence

class ArkheSolver:
    """
    Solver for quantum evolution with Arkhe(n) coherence coupling.
    """

    def __init__(self, H, c_ops=None, phi_coupling=0.05, phi_threshold=0.847, options=None):
        self.H = H
        self.c_ops = c_ops if c_ops else []
        self.phi_coupling = phi_coupling
        self.phi_threshold = phi_threshold
        self.options = options or {}
        self.L0 = liouvillian(H, self.c_ops)

    def _build_liouvillian(self, rho):
        """
        Build full Liouvillian including Φ coupling term.
        """
        phi_value = compute_phi(rho)
        phi_correction = 1.0 + self.phi_coupling * (phi_value - self.phi_threshold)
        return phi_correction * self.L0

    def solve(self, rho0, tlist, e_ops=None, track_coherence=True):
        """
        Solve time evolution with state-dependent Liouvillian.
        """
        from qutip import expect

        states = [rho0]
        coherence_history = []

        rho = rho0
        if rho.isket:
            rho = rho * rho.dag()

        dt = tlist[1] - tlist[0] if len(tlist) > 1 else 0

        for i in range(len(tlist)):
            t = tlist[i]
            phi_val = compute_phi(rho)

            if track_coherence:
                coherence_history.append({'time': t, 'phi': phi_val})

            if i == len(tlist) - 1:
                break

            L = self._build_liouvillian(rho)
            res = mesolve(L, rho, [0, dt])
            rho = res.states[-1]
            states.append(rho)

        # Wrap result
        arkhe_result = ArkheResult(states, tlist)
        arkhe_result.coherence_history = coherence_history

        if e_ops:
            if not isinstance(e_ops, list):
                e_ops = [e_ops]
            for op in e_ops:
                arkhe_result.expect.append([expect(op, s) for s in states])

        return arkhe_result

class ArkheResult:
    """
    Result class for ArkheSolver.
    """
    def __init__(self, states, times):
        self.states = states
        self.times = times
        self.expect = []
        self.coherence_history = []
        self.final_state = states[-1]
