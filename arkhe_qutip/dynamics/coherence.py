"""
Coherence metrics for Arkhe(n)-QuTiP.
Implements C (local coherence), Φ (integrated information),
and z (criticality parameter).
"""

import numpy as np
from qutip import Qobj, entropy_vn

def compute_purity(rho):
    """Compute purity of a quantum state: Tr(rho^2)."""
    return np.real((rho * rho).tr())

def compute_local_coherence(rho):
    """
    Compute local coherence C of a quantum state.

    Parameters
    ----------
    rho : Qobj
        Density matrix.

    Returns
    -------
    C : float
        Coherence in [0,1] (1 = pure, 0 = maximally mixed).
    """
    # Coherence as purity
    p = compute_purity(rho)
    # Map from [1/d, 1] to [0,1]
    d = rho.shape[0]
    return (p - 1/d) / (1 - 1/d) if d > 1 else p

def compute_fluctuation(rho):
    """
    Compute fluctuation F = 1 - C.

    Parameters
    ----------
    rho : Qobj
        Density matrix.

    Returns
    -------
    F : float
        Fluctuation in [0,1].
    """
    return 1 - compute_local_coherence(rho)

def compute_phi(rho):
    """
    Compute integrated information Φ (simplified).

    This is a placeholder for a proper IIT calculation.
    In practice, would require partitioning the system and
    computing mutual information between parts.

    Parameters
    ----------
    rho : Qobj
        Density matrix.

    Returns
    -------
    phi : float
        Approximate integrated information.
    """
    # For single systems, Φ is approximated by coherence
    return compute_local_coherence(rho)

def compute_z(rho):
    """
    Compute criticality parameter z.

    z is related to the instability of the system,
    approximated here as the ratio of fluctuation to coherence.

    Parameters
    ----------
    rho : Qobj
        Density matrix.

    Returns
    -------
    z : float
        Criticality parameter.
    """
    C = compute_local_coherence(rho)
    F = 1 - C
    if C == 0:
        return float('inf')
    return F / C

def check_critical_zone(z, phi=0.618, tolerance=0.2):
    """
    Check if system is in critical zone (z ≈ φ).

    Parameters
    ----------
    z : float
        Criticality parameter.
    phi : float, default=0.618
        Golden ratio threshold.
    tolerance : float, default=0.2
        Relative tolerance.

    Returns
    -------
    in_zone : bool
        True if |z - φ|/φ < tolerance.
    zone_name : str
        'UNDER_CRITICAL', 'CRITICAL', or 'OVER_CRITICAL'.
    """
    rel_diff = abs(z - phi) / phi
    if rel_diff < tolerance:
        return True, 'CRITICAL'
    elif z < phi:
        return False, 'UNDER_CRITICAL'
    else:
        return False, 'OVER_CRITICAL'
