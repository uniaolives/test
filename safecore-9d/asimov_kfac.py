# safecore-9d/asimov_kfac.py
# K-FAC Engine (EFIM Engine)
# Kronecker-Factored Approximate Curvature for Distributed Discovery

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
from functools import partial

@struct.dataclass
class KFACState:
    """Maintains the distributed Kronecker factors for the EFIM."""
    factor_a: jnp.ndarray  # Covariance of input activations
    factor_g: jnp.ndarray  # Covariance of output gradients
    step_count: int
    ema_decay: float = 0.95

class KFACEngine:
    """
    Kronecker-Factored Approximate Curvature (K-FAC) engine.
    Reduces Fisher inversion cost from O(d^3) to O(d^1.5).
    """
    def __init__(self, damping: float = 1e-3):
        self.damping = damping

    def init_state(self, dim_in: int, dim_out: int) -> KFACState:
        return KFACState(
            factor_a=jnp.eye(dim_in),
            factor_g=jnp.eye(dim_out),
            step_count=0
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_factors(self, state: KFACState, a: jnp.ndarray, g: jnp.ndarray) -> KFACState:
        """Updates the Empirical Fisher factors using an Exponential Moving Average."""
        # a: [batch, dim_in], g: [batch, dim_out]
        new_a = (state.ema_decay * state.factor_a +
                 (1 - state.ema_decay) * jnp.matmul(a.T, a) / a.shape[0])
        new_g = (state.ema_decay * state.factor_g +
                 (1 - state.ema_decay) * jnp.matmul(g.T, g) / g.shape[0])

        return state.replace(factor_a=new_a, factor_g=new_g, step_count=state.step_count + 1)

    @partial(jax.jit, static_argnums=(0,))
    def precondition(self, state: KFACState, gradient: jnp.ndarray) -> jnp.ndarray:
        """Inverts the factors and pre-multiplies the gradient (The Geodesic Step)."""
        inv_a = jnp.linalg.inv(state.factor_a + jnp.sqrt(self.damping) * jnp.eye(state.factor_a.shape[0]))
        inv_g = jnp.linalg.inv(state.factor_g + jnp.sqrt(self.damping) * jnp.eye(state.factor_g.shape[0]))
        return jnp.matmul(inv_g, jnp.matmul(gradient, inv_a))

    def get_curvature_metrics(self, state: KFACState):
        """Analyzes eigenvalues to monitor manifold stability."""
        ev_a = jnp.linalg.eigvalsh(state.factor_a)
        ev_g = jnp.linalg.eigvalsh(state.factor_g)
        return {
            "max_ev_a": float(jnp.max(ev_a)),
            "min_ev_a": float(jnp.min(ev_a)),
            "max_ev_g": float(jnp.max(ev_g)),
            "min_ev_g": float(jnp.min(ev_g)),
            "condition_number_g": float(jnp.max(ev_g) / (jnp.min(ev_g) + 1e-10))
        }
