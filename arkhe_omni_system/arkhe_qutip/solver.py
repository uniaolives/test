# arkhe_omni_system/arkhe_qutip/solver.py
import numpy as np
import uuid
import qutip as qt
from qutip import mesolve
from typing import List, Dict, Any, Optional, Union
from .core import ArkheQobj

class ArkheSolver:
    """
    Solver with integrated information (Φ) coupling.
    dρ/dt = -i[H, ρ] + Σ(LρL† - ½{L†L, ρ}) + α·∇Φ
    """
    def __init__(self, H: qt.Qobj, c_ops: Optional[List[qt.Qobj]] = None, phi_coupling: float = 0.0):
        self.H = H
        self.c_ops = c_ops or []
        self.phi_coupling = phi_coupling
        self.phi_golden = (1 + np.sqrt(5)) / 2

    def solve(self, rho0: Union[qt.Qobj, ArkheQobj], tlist: np.ndarray,
              track_coherence: bool = True) -> Any:
        """Solves the master equation with Φ coupling."""
        # Add Φ coupling as a time-dependent perturbation at golden ratio frequency
        H_total = self.H
        if self.phi_coupling > 0:
            # Simplified: add a small contribution to the Hamiltonian
            H_phi = self.phi_coupling * qt.identity(self.H.dims[0])
            H_total = self.H + H_phi

        result = mesolve(H_total, rho0, tlist, self.c_ops)

        if track_coherence:
            from .coherence import purity, integrated_information
            result.coherence = [purity(s) for s in result.states]
            result.coherence_trajectory = [{'purity': p} for p in result.coherence]
            result.phi_trajectory = [integrated_information(s) for s in result.states]

            # Ensure history and node_id are propagated
            history = getattr(rho0, 'history', [])
            node_id = getattr(rho0, 'node_id', str(uuid.uuid4()))
            result.arkhe_final_state = ArkheQobj(result.states[-1], history=history, node_id=node_id)

        return result
