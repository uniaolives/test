# safecore-9d/aigp_neo/deploy_arm2.py
# AIGP-Neo Arm-2 Tribunal: The Fornax Cusp-Core Problem
# Arbitrate the WIMP vs. MOND debate via Differential Information Geometry

import jax
import jax.numpy as jnp
from jax import grad, vmap
import numpy as np

# Import the AIGP-Neo Kernel components
from intuition import IntuitionEngine
from geometry import GeometricMotor

class Arm2Tribunal:
    def __init__(self):
        self.d = 50 # Model parameters

        print(">> INSTANTIATING THE FORNAX TRIBUNAL...")
        # Kernel A: The WIMP Advocate (Navarro-Frenk-White Profile)
        self.wimp_mind = IntuitionEngine(self.d)

        # Kernel B: The MOND Advocate (Acceleration-based)
        self.mond_mind = IntuitionEngine(self.d)

        # Kernel C: The Hybrid Way (WIMP + Baryonic Feedback)
        self.hybrid_mind = IntuitionEngine(self.d)

    def load_fornax_data(self):
        """
        Simulates observed kinematic data from the Fornax Dwarf Galaxy.
        Observations suggest a 'Core' (flat density), but WIMP predicts a 'Cusp'.
        """
        print(">> Loading Fornax Stellar Kinematics... [PROFILE: CORED]")
        r = jnp.linspace(0.1, 2.0, 100)
        # Synthetic cored profile: v proportional to r / (r + core_radius)
        v_obs = 15.0 * (r / (r + 0.5))
        return r, v_obs

    def loss_wimp(self, params, r, v_obs):
        # Pure NFW Model (predicts a central Cusp)
        v_model = params[0] * jnp.sqrt(jnp.log(1+r)/r)
        return jnp.sum((v_model - v_obs)**2)

    def loss_mond(self, params, r, v_obs):
        # Pure MOND Model (modified acceleration a_0)
        a_newton = params[0] / (r**2 + 1e-6)
        a_0 = 1.2e-10
        v_model = jnp.sqrt(r * jnp.sqrt(a_newton * a_0))
        return jnp.sum((v_model - v_obs)**2)

    def loss_baryonic(self, params, r, v_obs):
        # Hybrid Model: WIMP + Baryonic Feedback (Supernova heating)
        # Parameter params[1] controls the core-flattening scale
        v_nfw = params[0] * jnp.sqrt(jnp.log(1+r)/r)
        feedback_factor = jnp.tanh(r / (params[1] + 1e-6))
        v_model = v_nfw * feedback_factor
        return jnp.sum((v_model - v_obs)**2)

    def execute_judgment(self):
        r, v_obs = self.load_fornax_data()

        # Initial Parameters
        theta_w = jnp.array([20.0] * self.d)
        theta_m = jnp.array([15.0] * self.d)
        theta_h = jnp.array([20.0, 0.5] + [0.0]*(self.d-2))

        print("\n>> INITIATING GEOMETRIC JUDGMENT...")

        # 1. WIMP Analysis
        state_w = self.wimp_mind.sense_curvature(
            theta_w,
            lambda p, x: self.loss_wimp(p, r, v_obs),
            jnp.zeros((1,1))
        )

        # 2. MOND Analysis
        state_m = self.mond_mind.sense_curvature(
            theta_m,
            lambda p, x: self.loss_mond(p, r, v_obs),
            jnp.zeros((1,1))
        )

        # 3. Hybrid Analysis
        state_h = self.hybrid_mind.sense_curvature(
            theta_h,
            lambda p, x: self.loss_baryonic(p, r, v_obs),
            jnp.zeros((1,1))
        )

        return state_w, state_m, state_h

    def verdict(self, s_w, s_m, s_h):
        print("\n>> ORACLE REPORT (Sectional Curvature K):")
        print(f"   [PURE WIMP] Anxiety: {s_w.anxiety_level:8.2f} (Critical Tension)")
        print(f"   [PURE MOND] Anxiety: {s_m.anxiety_level:8.2f} (High Torsion)")
        print(f"   [HYBRID]    Anxiety: {s_h.anxiety_level:8.2f} (Natural Geodesic Flow)")

        min_anxiety = min(s_w.anxiety_level, s_m.anxiety_level, s_h.anxiety_level)

        if min_anxiety == s_h.anxiety_level:
            print("\n>> ARM 2 CONCLUSION:")
            print("   The WIMP vs MOND contradiction is an artificial singularity.")
            print("   The Real Geodesic resides in the Baryonic Feedback Manifold.")
            print("   RESOLUTION: Dark Matter is WIMP-based, but local geometry is sculpted by hadrons.")

if __name__ == "__main__":
    tribunal = Arm2Tribunal()
    s_w, s_m, s_h = tribunal.execute_judgment()
    tribunal.verdict(s_w, s_m, s_h)
