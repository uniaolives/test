# arkhe_qutip/core.py
import time
import uuid
import numpy as np
import qutip as qt
from qutip import Qobj, mesolve
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

@dataclass
class HandoverEvent:
    operator: qt.Qobj
    coherence_before: float
    coherence_after: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"HandoverEvent(op_type={self.metadata.get('type', 'unknown')}, C: {self.coherence_before:.3f} -> {self.coherence_after:.3f})"

class ArkheQobj(Qobj):
    """
    Extends QuTiP's Qobj with handover history, coherence tracking and node identity.
    """
    def __init__(self, inpt, *args, **kwargs):
        # Handle the case where we're creating from another ArkheQobj
        history = kwargs.pop('history', [])
        node_id = kwargs.pop('node_id', str(uuid.uuid4()))
        creation_time = kwargs.pop('creation_time', time.time())

        super().__init__(inpt, *args, **kwargs)

        self._history = history
        self._node_id = node_id
        self._creation_time = creation_time

    @property
    def history(self) -> List[HandoverEvent]:
        return self._history

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def coherence(self) -> float:
        """Automatic coherence calculation (purity)."""
        from .coherence import purity
        return purity(self)

    def handover(self, operator: qt.Qobj, metadata: Optional[Dict[str, Any]] = None) -> 'ArkheQobj':
        """Applies a quantum operator and records the handover event."""
        c_before = self.coherence

        # Apply operator: ρ' = O ρ O† or |ψ'⟩ = O |ψ⟩
        if self.isoper or self.isoperket or self.isoperbra:
             # If it's an operator, we might be composing them?
             # But usually handover is applied to a state.
             pass

        new_qobj = operator * self
        if self.isoper and operator.isoper:
             # If both are operators, it's composition
             pass
        elif self.isbra and operator.isoper:
             # bra * oper
             pass

        # For simplicity, handle most common case: operator * state
        # If self is a density matrix (isoper) and operator is unitary:
        if self.isoper and operator.isoper:
            if operator.isunitary:
                new_data = operator * self * operator.dag()
            else:
                # General map
                new_data = operator * self * operator.dag() # Simplified
        else:
            new_data = operator * self

        c_after = ArkheQobj(new_data).coherence

        event = HandoverEvent(
            operator=operator,
            coherence_before=c_before,
            coherence_after=c_after,
            metadata=metadata or {}
        )

        new_history = self._history + [event]

        return ArkheQobj(
            new_data,
            history=new_history,
            node_id=self._node_id,
            creation_time=self._creation_time
        )

    def get_coherence_trajectory(self) -> List[float]:
        """Returns the history of coherence values."""
        trajectory = []
        if self._history:
            trajectory = [h.coherence_before for h in self._history]
            trajectory.append(self._history[-1].coherence_after)
        else:
            trajectory = [self.coherence]
        return trajectory

    def evolve_with_handover(self, H: qt.Qobj, tlist: np.ndarray,
                             handovers: List[Tuple[float, qt.Qobj, Dict[str, Any]]],
                             c_ops: Optional[List[qt.Qobj]] = None) -> Tuple[List['ArkheQobj'], Any]:
        """Evolves the state with scheduled handovers at specific times."""
        current_state = self
        all_states = []

        sorted_handovers = sorted(handovers, key=lambda x: x[0])
        handover_idx = 0

        last_t = tlist[0]
        all_states.append(current_state)

        for t in tlist[1:]:
            # Check for handovers between last_t and t
            while handover_idx < len(sorted_handovers) and sorted_handovers[handover_idx][0] <= t:
                h_time, h_op, h_meta = sorted_handovers[handover_idx]
                # Evolve to handover time
                if h_time > last_t:
                    res = mesolve(H, current_state, [last_t, h_time], c_ops or [])
                    current_state = ArkheQobj(res.states[-1], history=current_state.history, node_id=self._node_id)
                # Apply handover
                current_state = current_state.handover(h_op, h_meta)
                last_t = h_time
                handover_idx += 1

            # Evolve to t
            if t > last_t:
                res = mesolve(H, current_state, [last_t, t], c_ops or [])
                current_state = ArkheQobj(res.states[-1], history=current_state.history, node_id=self._node_id)
                last_t = t

            all_states.append(current_state)

        return all_states, None
