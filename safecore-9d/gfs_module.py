# safecore-9d/gfs_module.py
# GFS Module: Greedy Forward Selection
# Enforces "1.4% sparsity" rule by selecting the most informative geodesics

import jax
import jax.numpy as jnp
from asimov_kfac import KFACState

class GFSModule:
    """
    Identifies the 'Sparse Backbone' of the information manifold.
    Prunes gradients of non-essential parameters to maintain biological-grade sparsity.
    """
    def __init__(self, target_sparsity: float = 0.014):
        self.target_sparsity = target_sparsity

    def select_backbone(self, gradient: jnp.ndarray, kfac_state: KFACState) -> jnp.ndarray:
        """
        Ranks parameters based on sensitivity score and prunes low-importance ones.

        I_j = |g_j| * sqrt(diag(F^-1)_j)
        """
        # gradient: [dim_out, dim_in]
        # Approximation of diagonal of inverse Fisher from Kronecker factors
        # diag(A ⊗ G)^-1 = diag(A^-1) ⊗ diag(G^-1)

        diag_a_inv = jnp.diag(jnp.linalg.inv(kfac_state.factor_a + 1e-4 * jnp.eye(kfac_state.factor_a.shape[0])))
        diag_g_inv = jnp.diag(jnp.linalg.inv(kfac_state.factor_g + 1e-4 * jnp.eye(kfac_state.factor_g.shape[0])))

        # Outer product of diagonals gives diagonal of the full Kronecker product inverse
        diag_f_inv = jnp.outer(diag_g_inv, diag_a_inv) # [dim_out, dim_in]

        # Sensitivity Score (Information Importance)
        importance = jnp.abs(gradient) * jnp.sqrt(diag_f_inv)

        # Determine number of parameters to keep
        num_params = gradient.size
        k = max(1, int(num_params * self.target_sparsity))

        # Flatten and find threshold for top k
        flat_importance = importance.flatten()
        threshold = jnp.sort(flat_importance)[-k]

        # Create sparse mask
        mask = (importance >= threshold).astype(jnp.float32)

        # Return pruned gradient
        return gradient * mask, importance

    def get_sparsity_report(self, importance: jnp.ndarray):
        """Generates a summary of the information distribution."""
        return {
            "mean_importance": jnp.mean(importance),
            "max_importance": jnp.max(importance),
            "sparsity_ratio": self.target_sparsity,
            "entropy": -jnp.sum(importance * jnp.log(importance + 1e-10)) / importance.size
        }
