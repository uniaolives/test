import numpy as np
from .muzero import Field, Handover

def intrinsic_reward(state: Field, action: Handover, next_state: Field) -> float:
    """
    Reward = entropy reduction (coherence gain).
    """
    entropy_before = state.entropy()
    entropy_after = next_state.entropy()
    coherence_gain = entropy_before - entropy_after

    if new_invariant_detected(state, next_state):
        coherence_gain += 10.0

    return coherence_gain

def new_invariant_detected(before: Field, after: Field) -> bool:
    """
    Mock detector for physical invariants (e.g., F1*d1 = F2*d2).
    """
    # Procura por relações invariantes simuladas
    return False

class PhysicalLaw:
    def __init__(self, name: str, formula_lambda):
        self.name = name
        self.formula = formula_lambda
        self.verification_count = 0

    def verify(self, field: Field) -> bool:
        self.verification_count += 1
        return self.formula(field)
