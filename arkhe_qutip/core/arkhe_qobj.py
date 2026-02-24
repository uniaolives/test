"""
ArkheQobj: Quantum Object with Handover Memory
Extends QuTiP's Qobj to include:
- Handover history (log of interactions)
- Local coherence C
- Fluctuation F (with C + F = 1)
- Winding number (topological invariant)
"""

import numpy as np
from qutip import Qobj
import time
import hashlib

class ArkheQobj(Qobj):
    """
    Quantum object with Arkhe(n) attributes.

    Parameters
    ----------
    arg : array_like or Qobj
        Input data for quantum object.
    dims : list, optional
        Dimensions of quantum object.
    copy : bool, optional
        Whether to copy input data.
    node_id : str, optional
        Unique identifier for this node. If None, generated from hash.
    """

    def __init__(self, arg, dims=None, copy=True, node_id=None):
        super().__init__(arg, dims=dims, copy=copy)

        # Identity
        self.node_id = node_id or self._generate_id()

        # Handover history
        self.handover_log = []  # list of (timestamp, handover_id, target_id, metadata)

        # Coherence metrics
        self._coherence = 1.0  # C_local (initialized as maximum)
        self._fluctuation = 0.0  # F_local (initialized as minimum)
        # Note: C + F = 1 is enforced by setters

        # Topological invariants
        self.winding_number = 0
        self.accumulated_phase = 1.0 + 0.0j  # complex phase from braiding

        # Timestamp of last update
        self.last_update = time.time()

    def _update_state(self, new_qobj):
        """
        Update the internal Qobj state while preserving Arkhe attributes.
        WARNING: This modifies internal Qobj attributes (_data, _dims) directly,
        bypassing immutability. Use with caution.
        """
        self._data = new_qobj.data
        if hasattr(new_qobj, '_dims'):
            self._dims = new_qobj._dims
        else:
            self.dims = new_qobj.dims
        self.last_update = time.time()

    def _generate_id(self):
        """Generate unique node ID from object data."""
        data_hash = hashlib.sha256(str(self.data).encode()).hexdigest()[:16]
        return f"arkhe_node_{data_hash}"

    @property
    def coherence(self):
        """Local coherence C (read-only)."""
        return self._coherence

    @coherence.setter
    def coherence(self, value):
        """Set coherence, automatically adjusting fluctuation to maintain C + F = 1."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Coherence must be in [0, 1]")
        self._coherence = value
        self._fluctuation = 1.0 - value

    @property
    def fluctuation(self):
        """Local fluctuation F (read-only)."""
        return self._fluctuation

    def update_coherence(self, delta_c):
        """
        Update coherence by delta_c, maintaining C + F = 1.

        Parameters
        ----------
        delta_c : float
            Change in coherence (positive or negative).
        """
        new_c = np.clip(self.coherence + delta_c, 0, 1)
        self.coherence = new_c
        return self.coherence

    def record_handover(self, handover_id, target_id, metadata=None):
        """
        Record a handover in the node's history.

        Parameters
        ----------
        handover_id : str
            Identifier of the handover.
        target_id : str
            Identifier of the target node.
        metadata : dict, optional
            Additional metadata about the handover.
        """
        entry = {
            'timestamp': time.time(),
            'handover_id': handover_id,
            'target_id': target_id,
            'metadata': metadata or {}
        }
        self.handover_log.append(entry)

        # Update winding number (simplified: increment for each handover)
        self.winding_number += 1

    def apply_phase(self, phase):
        """
        Apply a complex phase to the node (accumulated from braiding).

        Parameters
        ----------
        phase : complex
            Phase factor e^{iθ} to multiply.
        """
        self.accumulated_phase *= phase

        # Normalize to unit magnitude (keep as phase)
        self.accumulated_phase /= abs(self.accumulated_phase)

    def copy(self):
        """Create a copy of this ArkheQobj."""
        new = ArkheQobj(self.data, dims=self.dims, node_id=self.node_id + "_copy")
        new.handover_log = self.handover_log.copy()
        new._coherence = self._coherence
        new._fluctuation = self._fluctuation
        new.winding_number = self.winding_number
        new.accumulated_phase = self.accumulated_phase
        return new

    def __repr__(self):
        lines = super().__repr__().split('\n')
        lines.insert(1, f"Node ID: {self.node_id}")
        lines.insert(2, f"Coherence: {self.coherence:.4f}, Fluctuation: {self.fluctuation:.4f}")
        lines.insert(3, f"Winding number: {self.winding_number}")
        return '\n'.join(lines)
