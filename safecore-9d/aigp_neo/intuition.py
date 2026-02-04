# safecore-9d/aigp_neo/intuition.py
# AIGP-Neo: Intuition Engine
# Sense the manifold 'texture' to detect anxiety (curvature) or confidence

import jax
import jax.numpy as jnp
from jax import vmap, jit
from dataclasses import dataclass

@dataclass
class IntuitionState:
    anxiety_level: float  # Correlates with curvature intensity
    confidence: float     # Correlates with Fisher metric stability
    focus_mode: str       # "EXPLORE" vs "CONSOLIDATE"

class IntuitionEngine:
    def __init__(self, param_dim: int):
        self.d = param_dim

    def sense_curvature(self, params, loss_fn, batch_data):
        """
        Samples the 'texture' of the manifold.
        High sectional curvature = 'Rough' terrain (Hidden Instability).
        """
        # Stochastic Geodesic Deviation
        key = jax.random.PRNGKey(42)
        v = jax.random.normal(key, params.shape)
        v /= (jnp.linalg.norm(v) + 1e-10)

        # Approximate Natural Gradient (using diagonal Fisher)
        # Compute gradients for each observation in the batch
        grads = vmap(jax.grad(loss_fn), in_axes=(None, 0))(params, batch_data)
        fisher_diag = jnp.mean(grads**2, axis=0) + 1e-6

        # Sensitivity to state perturbation
        ng_v = v / fisher_diag
        curvature_intensity = jnp.linalg.norm(ng_v)

        return IntuitionState(
            anxiety_level=float(curvature_intensity),
            confidence=1.0 / (1.0 + float(jnp.var(grads))),
            focus_mode="CONSOLIDATE" if curvature_intensity > 1e3 else "EXPLORE"
        )
