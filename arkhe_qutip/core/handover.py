"""
Quantum Handover: Interaction between ArkheQobj nodes.
Implements the handover concept from Arkhe(n) Language in quantum domain.
"""

import numpy as np
import time
from qutip import Qobj, tensor, identity
from .arkhe_qobj import ArkheQobj

class QuantumHandover:
    """
    Quantum handover between two ArkheQobj nodes.

    A handover represents an interaction that transfers quantum information,
    modifies states, and updates coherence metrics.

    Parameters
    ----------
    handover_id : str
        Unique identifier for this handover.
    source : ArkheQobj
        Source node (initiates the handover).
    target : ArkheQobj
        Target node (receives the handover).
    operator : Qobj, optional
        Quantum operator representing the interaction.
        If None, a swap-like interaction is assumed.
    metadata : dict, optional
        Additional metadata about the handover.
    """

    # Protocol types (from ANL)
    PROTOCOL_CONSERVATIVE = 1  # Preserves information (e.g., unitary)
    PROTOCOL_CREATIVE = 2       # Generates new information (e.g., measurement)
    PROTOCOL_DESTRUCTIVE = 3    # Removes information (e.g., dissipation)
    PROTOCOL_TRANSMUTATIVE = 4  # Changes type (e.g., ket to density matrix)

    def __init__(self, handover_id, source, target, operator=None, metadata=None):
        self.handover_id = handover_id
        self.source = source
        self.target = target
        self.operator = operator
        self.metadata = metadata or {}
        self.timestamp = None
        self.result = None
        self.success = False

        # Determine protocol type based on operator (heuristic)
        self.protocol = self._infer_protocol()

    def _infer_protocol(self):
        """Infer handover protocol type from operator properties."""
        if self.operator is None:
            return self.PROTOCOL_CONSERVATIVE

        if self.operator.isherm:
            # Hermitian operators often correspond to measurements
            return self.PROTOCOL_CREATIVE
        elif self.operator.isunitary:
            # Unitary operators are conservative
            return self.PROTOCOL_CONSERVATIVE
        else:
            # Non-unitary, non-Hermitian could be dissipative
            return self.PROTOCOL_DESTRUCTIVE

    def execute(self):
        """
        Execute the handover.

        Returns
        -------
        result : Qobj or None
            Result of the handover (e.g., measurement outcome, new state).
        """
        self.timestamp = time.time()

        # Create combined system if operator acts on both
        if self.operator is not None:
            if self.source is self.target:
                # Self-handover: operator acts on single node
                new_state = self.operator * self.source

                # Update source/target in place to preserve Arkhe attributes
                self.source._update_state(new_state)

                self.result = self.source
                self.success = True
            else:
                # Interaction between two nodes
                combined = tensor(self.source, self.target)
                result_state = self.operator * combined

                # Update states in place to preserve Arkhe attributes
                # Update target state (trace out source)
                self.target._update_state(result_state.ptrace(1))

                # Update source state (trace out target)
                self.source._update_state(result_state.ptrace(0))

                self.result = result_state
                self.success = True
        else:
            # Default swap-like interaction
            # Actually swap the internal states of the nodes
            source_state = Qobj(self.source.data, dims=self.source.dims)
            target_state = Qobj(self.target.data, dims=self.target.dims)

            self.source._update_state(target_state)
            self.target._update_state(source_state)

            self.result = self.target
            self.success = True

        # Record handover in both nodes
        self.source.record_handover(self.handover_id, self.target.node_id, self.metadata)
        self.target.record_handover(self.handover_id, self.source.node_id, self.metadata)

        # Update coherence based on protocol
        self._update_coherence()

        return self.result

    def _update_coherence(self):
        """Update node coherences based on handover protocol."""
        if self.protocol == self.PROTOCOL_CONSERVATIVE:
            # Conservative handover preserves coherence
            pass
        elif self.protocol == self.PROTOCOL_CREATIVE:
            # Creative handover increases fluctuation (decreases coherence)
            self.source.update_coherence(-0.05)
            self.target.update_coherence(-0.05)
        elif self.protocol == self.PROTOCOL_DESTRUCTIVE:
            # Destructive handover significantly reduces coherence
            self.source.update_coherence(-0.1)
            self.target.update_coherence(-0.1)
        elif self.protocol == self.PROTOCOL_TRANSMUTATIVE:
            # Transmutative handover changes type - reset coherence
            self.source.coherence = 0.5
            self.target.coherence = 0.5

    def __repr__(self):
        status = "SUCCESS" if self.success else "PENDING"
        return (f"QuantumHandover(id={self.handover_id}, "
                f"source={self.source.node_id[:8]}..., "
                f"target={self.target.node_id[:8]}..., "
                f"protocol={self.protocol}, status={status})")
