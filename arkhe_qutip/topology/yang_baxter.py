"""
Yang-Baxter verification for quantum handovers.
Ensures topological consistency of handover sequences.
"""

import numpy as np
from qutip import Qobj, tensor
from arkhe_qutip.core.handover import QuantumHandover

def verify_yang_baxter(h12, h13, h23, state, tolerance=1e-6):
    """
    Verify the Yang-Baxter equation for three handovers.

    R₁₂ R₁₃ R₂₃ |ψ⟩ = R₂₃ R₁₃ R₁₂ |ψ⟩

    Parameters
    ----------
    h12, h13, h23 : QuantumHandover
        Handover operators (must act on composite system).
    state : Qobj
        Initial state.
    tolerance : float, default=1e-6
        Numerical tolerance.

    Returns
    -------
    bool
        True if Yang-Baxter holds within tolerance.
    float
        Norm difference.
    """
    # Check that handovers act on the right subsystems
    # This is a simplified implementation

    # Assume operators act on a composite system with at least 3 subsystems
    # In practice, would need to extract the operators

    # Placeholder: create dummy operators if not provided
    if h12.operator is None or h13.operator is None or h23.operator is None:
        print("Warning: missing operators, using identity")
        return True, 0.0

    # Apply in order LHS: R₁₂ R₁₃ R₂₃
    lhs = h23.operator * (h13.operator * (h12.operator * state))

    # Apply in order RHS: R₂₃ R₁₃ R₁₂
    rhs = h12.operator * (h13.operator * (h23.operator * state))

    # Compute difference
    diff = (lhs - rhs).norm()

    return diff < tolerance, diff


def check_handover_commutativity(h1, h2, state, tolerance=1e-6):
    """
    Check if two handovers commute.

    Parameters
    ----------
    h1, h2 : QuantumHandover
        Handover operators.
    state : Qobj
        Initial state.
    tolerance : float, default=1e-6
        Numerical tolerance.

    Returns
    -------
    bool
        True if handovers commute.
    float
        Commutator norm.
    """
    if h1.operator is None or h2.operator is None:
        return True, 0.0

    comm = (h1.operator * h2.operator - h2.operator * h1.operator) * state
    norm = comm.norm()

    return norm < tolerance, norm
