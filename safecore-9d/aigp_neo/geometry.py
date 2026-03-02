# safecore-9d/aigp_neo/geometry.py
# AIGP-Neo: Geometric Motor
# Executes movement along information geodesics (Natural Gradient)

import jax.numpy as jnp
from intuition import IntuitionState

class GeometricMotor:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, params, grads, fisher_diag, intuition: IntuitionState):
        """
        Executes a movement step modulated by intuition.
        """
        # Modulation: If anxious (high curvature), slow down to maintain stability
        adaptive_lr = self.lr / (1.0 + 0.1 * intuition.anxiety_level)

        # Natural Gradient Step: d_theta = F^-1 * grad
        # Moves a constant distance in Information Space
        natural_grad = grads / (fisher_diag + 1e-6)

        # Update parameters
        new_params = params - adaptive_lr * natural_grad
        return new_params
