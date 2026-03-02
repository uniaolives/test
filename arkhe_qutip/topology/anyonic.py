"""
Anyonic statistics for Arkhe(n)-QuTiP.
Implements fractional statistics and braiding operations.
"""

import numpy as np
from qutip import Qobj, tensor, identity

class AnyonStatistic:
    """
    Anyonic statistic parameter α ∈ [0,1].

    α = 0 → bosonic (symmetric)
    α = 1 → fermionic (antisymmetric)
    0 < α < 1 → anyonic (fractional)
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : float
            Anyonic parameter in [0,1].
        """
        if not 0 <= alpha <= 1:
            raise ValueError("α must be in [0,1]")
        self.alpha = alpha

    @property
    def phase(self):
        """Braiding phase e^{iπα}."""
        return np.exp(1j * np.pi * self.alpha)

    @property
    def is_bosonic(self):
        return self.alpha == 0

    @property
    def is_fermionic(self):
        return self.alpha == 1

    @property
    def is_anyonic(self):
        return 0 < self.alpha < 1

    def exchange_phase(self, other):
        """
        Exchange phase between two anyons.

        For identical anyons: e^{iπα}
        For different anyons: e^{iπ(α₁+α₂)/2}

        Parameters
        ----------
        other : AnyonStatistic
            Other anyon's statistic.

        Returns
        -------
        phase : complex
            Exchange phase factor.
        """
        avg_alpha = (self.alpha + other.alpha) / 2
        return np.exp(1j * np.pi * avg_alpha)

    def __repr__(self):
        return f"AnyonStatistic(α={self.alpha:.3f})"


class AnyonNode:
    """
    Anyonic node with fractional statistics.
    Extends ArkheQobj with anyonic properties.
    """

    def __init__(self, state, statistic, node_id=None):
        """
        Parameters
        ----------
        state : Qobj
            Quantum state (ket or density matrix).
        statistic : AnyonStatistic
            Anyonic statistic.
        node_id : str, optional
            Node identifier.
        """
        from arkhe_qutip.core.arkhe_qobj import ArkheQobj
        self.qobj = ArkheQobj(state, node_id=node_id)
        self.statistic = statistic
        self.braid_history = []  # list of (other_id, phase)

    def braid_with(self, other):
        """
        Perform braiding operation with another anyon.

        Parameters
        ----------
        other : AnyonNode
            Other anyon to braid with.

        Returns
        -------
        phase : complex
            Accumulated phase from braiding.
        """
        phase = self.statistic.exchange_phase(other.statistic)

        # Apply phase to both nodes
        self.qobj.apply_phase(phase)
        other.qobj.apply_phase(np.conj(phase))  # conjugate for other

        # Record history
        self.braid_history.append((other.qobj.node_id, phase))
        other.braid_history.append((self.qobj.node_id, np.conj(phase)))

        return phase

    @property
    def node_id(self):
        return self.qobj.node_id

    @property
    def coherence(self):
        return self.qobj.coherence

    def __repr__(self):
        return f"AnyonNode({self.node_id[:8]}..., α={self.statistic.alpha:.3f})"


def create_anyon_braid_circuit(anyons, braiding_sequence):
    """
    Create a quantum circuit representing a braiding sequence of anyons.

    Parameters
    ----------
    anyons : list of AnyonNode
        List of anyons.
    braiding_sequence : list of tuples (i, j)
        Sequence of braiding operations between anyons i and j.

    Returns
    -------
    circuit : Qobj
        Unitary representing the total braiding operation.
    """
    from qutip.qip.circuit import QubitCircuit

    n = len(anyons)
    qc = QubitCircuit(n)

    # Add gates corresponding to braiding operations
    for i, j in braiding_sequence:
        # Braiding between anyons i and j corresponds to a swap gate
        # with a phase factor depending on statistics
        phase = anyons[i].statistic.exchange_phase(anyons[j].statistic)

        # Create controlled-phase gate (simplified)
        # In practice, braiding is more complex
        qc.add_gate("SWAP", targets=[i, j])
        qc.add_gate("CPHASE", targets=[i, j], arg_value=np.angle(phase))

    return qc
