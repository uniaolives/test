"""
ArkheSolver: Time evolution with Φ coupling.
Extends QuTiP's master equation solvers to include
the Arkhe(n) coherence coupling term.
"""

import numpy as np
from qutip import mesolve, Qobj, liouvillian, lindblad_dissipator
from .coherence import compute_phi, compute_local_coherence
from qutip.solver import Result
from .coherence import compute_phi

class ArkheSolver:
    """
    Solver for quantum evolution with Arkhe(n) coherence coupling.
    """

    The evolution follows a modified Lindblad master equation:
    dρ/dt = -i[H,ρ] + Σ(L_i ρ L_i† - ½{L_i†L_i, ρ}) + α_φ · φ · [Φ, ρ]

    where Φ is the integrated information operator.

    Parameters
    ----------
    H : Qobj
        System Hamiltonian.
    c_ops : list of Qobj
        Collapse operators (Lindblad operators).
    phi_coupling : float, default=0.05
        Coupling strength α_φ for the Φ term.
    phi_threshold : float, default=0.847
        Critical threshold Ψ for coherence.
    options : dict, optional
        Solver options (passed to mesolve).
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
        # Build Liouvillian without Φ term (base Lindblad)
        self.L0 = liouvillian(H, self.c_ops)

        # Φ operator (simplified - in practice would need to compute from state)
        # This is a placeholder - real implementation would need adaptive Φ calculation
        self.Phi = None  # Will be computed during evolution

    def _build_liouvillian(self, rho):
        """
        Build full Liouvillian including Φ coupling term.

        The Φ term acts as a non-linear feedback that modifies the
        system's evolution based on its current integrated information (coherence).
        """
        # Compute Φ from current state
        phi_value = compute_phi(rho)

        # Non-linear coupling: Φ modifies the effective Hamiltonian or dissipation
        # Here we model it as an enhancement/damping of the coherent evolution
        # based on the Φ value.
        phi_correction = 1.0 + self.phi_coupling * (phi_value - self.phi_threshold)

        # Modified Liouvillian
        return phi_correction * self.L0

    def solve(self, rho0, tlist, e_ops=None, track_coherence=True):
        """
        Solve time evolution with state-dependent Liouvillian.
        """
        from qutip import expect

        states = [rho0]
        coherence_history = []
        phi_history = []

        # Expectation values storage
        if e_ops:
            if not isinstance(e_ops, list):
                e_ops = [e_ops]
            expect_vals = [[] for _ in e_ops]
        else:
            expect_vals = []

        rho = rho0
        dt = tlist[1] - tlist[0] if len(tlist) > 1 else 0

        for i, t in enumerate(tlist):
            # Compute metrics for current state
            phi_val = compute_phi(rho)
            if track_coherence:
                purity_val = (rho * rho).tr()
                coherence_history.append({'time': t, 'purity': purity_val, 'phi': phi_val})

            # Compute expectation values
            for j, op in enumerate(e_ops or []):
                expect_vals[j].append(expect(op, rho))

            if i == len(tlist) - 1:
                break

            # Step forward using the full Liouvillian
            L = self._build_liouvillian(rho)

            # QuTiP superoperator evolution: rho(t+dt) = exp(L*dt) * rho(t)
            # For simplicity and speed in simulation, we use the first order expansion
            # Or use mesolve for a single step
            res = mesolve(L, rho, [0, dt], options=self.options)
            rho = res.states[-1]
            states.append(rho)

        # Wrap result
        from qutip.solver import Result
        q_res = Result()
        q_res.times = tlist
        q_res.states = states
        q_res.expect = expect_vals

        arkhe_result = ArkheResult(q_res)
        arkhe_result.coherence_history = coherence_history
        arkhe_result.final_state = states[-1]

        return arkhe_result


class ArkheResult:
    """
    Result class for ArkheSolver.
    Wraps QuTiP's Result and adds coherence/phi tracking.
    """

    def __init__(self, qutip_result):
        self.qutip_result = qutip_result
        self.coherence_history = []
        self.phi_history = []
        self.final_state = None

    @property
    def times(self):
        return self.qutip_result.times

    @property
    def expect(self):
        return self.qutip_result.expect

    @property
    def states(self):
        return self.qutip_result.states
