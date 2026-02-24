import numpy as np
import time
import hashlib
from qutip import Qobj, mesolve

class ArkheQobj(Qobj):
    """
    Quantum object with Arkhe(n) attributes.
    Extends QuTiP's Qobj to include handover history and coherence metrics.
    """

    def __init__(self, arg, dims=None, copy=True, node_id=None):
        # Support both Qobj and array_like
        if isinstance(arg, Qobj):
            super().__init__(arg.data, dims=arg.dims, copy=copy)
        else:
            super().__init__(arg, dims=dims, copy=copy)

        self.node_id = node_id or self._generate_id()
        self.handover_log = []
        self._coherence = 1.0
        self._fluctuation = 0.0
        self.winding_number = 0
        self.accumulated_phase = 1.0 + 0.0j
        self.last_update = time.time()

    def _generate_id(self):
        data_hash = hashlib.sha256(str(self.data).encode()).hexdigest()[:16]
        return f"arkhe_node_{data_hash}"

    @property
    def coherence(self):
        return self._coherence

    @coherence.setter
    def coherence(self, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Coherence must be in [0, 1]")
        self._coherence = value
        self._fluctuation = 1.0 - value

    @property
    def fluctuation(self):
        return self._fluctuation

    @property
    def history(self):
        return self.handover_log

    def update_coherence(self, delta_c):
        new_c = np.clip(self.coherence + delta_c, 0, 1)
        self.coherence = new_c
        return self.coherence

    def _update_state(self, new_qobj):
        """
        Internal method to update the quantum state in-place while
        preserving Arkhe identity and metadata.
        Necessary for long-lived agent identity in QuTiP.
        """
        if not isinstance(new_qobj, Qobj):
            new_qobj = Qobj(new_qobj)

        # Accessing protected attributes to bypass immutability for Arkhe persistence
        self._data = new_qobj.data
        self._dims = new_qobj.dims
        self.last_update = time.time()

    def record_handover(self, handover_id, target_id, metadata=None):
        class HandoverEvent:
            def __init__(self, h_id, t_id, meta):
                self.handover_id = h_id
                self.target_id = t_id
                self.metadata = meta or {}
                self.timestamp = time.time()

        event = HandoverEvent(handover_id, target_id, metadata)
        self.handover_log.append(event)
        self.winding_number += 1
        return event

    def handover(self, operator, metadata=None):
        """
        Apply an operator and return a NEW ArkheQobj with updated state and copied history.
        This follows the standard Qobj immutable pattern while propagating Arkhe metadata.
        """
        new_data = operator * self
        new_obj = ArkheQobj(new_data, node_id=self.node_id)
        new_obj.handover_log = self.handover_log.copy()
        new_obj._coherence = self._coherence
        new_obj._fluctuation = self._fluctuation
        new_obj.winding_number = self.winding_number
        new_obj.accumulated_phase = self.accumulated_phase

        new_obj.record_handover(f"h_{int(time.time())}", self.node_id, metadata)
        return new_obj

    def apply_phase(self, phase):
        self.accumulated_phase *= phase
        self.accumulated_phase /= abs(self.accumulated_phase)

    def copy(self):
        new = ArkheQobj(self, node_id=self.node_id + "_copy")
        new.handover_log = self.handover_log.copy()
        new._coherence = self._coherence
        new._fluctuation = self._fluctuation
        new.winding_number = self.winding_number
        new.accumulated_phase = self.accumulated_phase
        return new

    def evolve_with_handover(self, H, tlist, handovers):
        all_states = []
        current_psi = self

        for i in range(len(tlist)):
            t = tlist[i]
            if i > 0:
                dt = t - tlist[i-1]
                res = mesolve(H, current_psi, [0, dt], [])
                # Update current_psi in-place or return new one?
                # For this method, we want a trajectory of states.
                # Standard evolve returns new objects for each time step.
                current_psi = ArkheQobj(res.states[-1], node_id=current_psi.node_id)
                current_psi.handover_log = all_states[-1].handover_log.copy()

            for t_h, op, meta in handovers:
                if i > 0 and tlist[i-1] < t_h <= t:
                     current_psi = current_psi.handover(op, meta)

            all_states.append(current_psi.copy())

        return all_states, current_psi

    def __repr__(self):
        lines = super().__repr__().split('\n')
        lines.insert(1, f"Node ID: {self.node_id}")
        lines.insert(2, f"Coherence: {self.coherence:.4f}, Fluctuation: {self.fluctuation:.4f}")
        lines.insert(3, f"Winding number: {self.winding_number}")
        return '\n'.join(lines)
