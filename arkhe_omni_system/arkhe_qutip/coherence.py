# arkhe_qutip/coherence.py
import numpy as np
import qutip as qt
from typing import Dict, Any, List

def purity(rho: qt.Qobj) -> float:
    """Calculates the purity of a quantum state: Tr(ρ²)."""
    if rho.isket or rho.isbra:
        return 1.0
    rho_sq = rho * rho
    return np.real(rho_sq.tr())

def von_neumann_entropy(rho: qt.Qobj) -> float:
    """Calculates the Von Neumann entropy: -Tr(ρ log ρ)."""
    return qt.entropy_vn(rho)

def coherence_l1(rho: qt.Qobj) -> float:
    """Calculates the l1-norm of coherence: Σ_{i≠j} |ρ_ij|."""
    if rho.isket:
        rho = qt.ket2dm(rho)
    elif rho.isbra:
        rho = qt.bra2dm(rho)

    data = rho.full()
    off_diag = data - np.diag(np.diag(data))
    return np.sum(np.abs(off_diag))

def integrated_information(state: qt.Qobj) -> float:
    """
    Simplified Integrated Information (Φ).
    Proxy based on l1-coherence and purity.
    """
    c_l1 = coherence_l1(state)
    p = purity(state)
    # Simplified heuristic for Φ
    return c_l1 * (2.0 - p)

def coherence_trajectory_analysis(trajectory: List[float]) -> Dict[str, Any]:
    """Analyzes a trajectory of coherence values."""
    if not trajectory:
        return {}

    diffs = np.diff(trajectory)
    trend = np.mean(diffs) if len(diffs) > 0 else 0.0

    return {
        'initial': trajectory[0],
        'final': trajectory[-1],
        'min': np.min(trajectory),
        'max': np.max(trajectory),
        'trend': "positive" if trend > 0.001 else "negative" if trend < -0.001 else "stable",
        'variance': np.var(trajectory)
    }
