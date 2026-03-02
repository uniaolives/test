# safecore-9d/aigp_neo/deploy_arm3.py
# AIGP-Neo Arm-3: Active Detection Protocol
# Mapping Arm 2 insights to physical targets for next-gen telescopes (JWST, Euclid, Roman).

import jax
import jax.numpy as jnp
from jax import grad, vmap
import numpy as np

# Import the AIGP-Neo Kernel components
from intuition import IntuitionEngine

class Arm3Navigator:
    def __init__(self):
        self.d = 50
        print(">> INITIALIZING OBSERVATION NAVIGATOR...")
        self.oracle = IntuitionEngine(self.d)

    def sensitivity_map(self, params, r_grid):
        """
        Calculates the expected 'Baryonic Signature' as a function of galactic radius.
        Identifies where the geometric difference between Pure WIMP and WIMP+Baryon is maximal.
        """
        # Pure Model (NFW)
        v_pure = params[0] * jnp.sqrt(jnp.log(1+r_grid)/(r_grid + 1e-6))

        # Hybrid Model (With Feedback)
        feedback_scale = params[1]
        v_hybrid = v_pure * jnp.tanh(r_grid / (feedback_scale + 1e-6))

        # The "Signature" is the spectral divergence between the two models
        signature = jnp.abs(v_pure - v_hybrid)
        return signature

    def target_acquisition(self):
        print(">> CALCULATING OPTIMAL POINTING VECTORS...")

        # Search Space: Dwarf Galaxies (r < 5 kpc)
        r_grid = jnp.linspace(0.01, 5.0, 500)

        # Physical parameters for an Ultra-Faint Dwarf (UFD)
        params = jnp.array([20.0, 0.5] + [0.0]*(self.d-2))

        # 1. Sensitivity Scan
        signature = self.sensitivity_map(params, r_grid)

        # Find the signature peak
        peak_idx = jnp.argmax(signature)
        r_optimal = r_grid[peak_idx]
        max_sig = signature[peak_idx]

        # 2. Curvature Validation
        mock_obs = signature[peak_idx] + jax.random.normal(jax.random.PRNGKey(0), shape=(1,))*0.1

        # Oracle senses if this observation reduces model anxiety
        # Fixed: batch_data must be a 1D array for vmap(jax.grad) to receive a scalar x
        intuition_state = self.oracle.sense_curvature(
            params,
            lambda p, x: (self.sensitivity_map(p, r_optimal) - x)**2,
            jnp.array([mock_obs[0]])
        )

        return r_optimal, max_sig, intuition_state

    def generate_observing_proposal(self, r_opt, sig, state):
        print("\n>> GENERATING OBSERVATION PROPOSAL (JWST / EUCLID / ROMAN)...")
        print("---------------------------------------------------------------")
        print(f"PRIMARY TARGET:  Central Region of Ultra-Faint Dwarf Galaxies")
        print(f"OPTIMAL RADIUS:  r = {r_opt:.3f} kpc (From galactic center)")
        print(f"EXPECTED SIGNAL: {sig:.2f} km/s (Circular velocity deviation)")
        print(f"AIGP CONFIDENCE: {state.confidence*100:.1f}% (Discrimination Probability)")
        print("---------------------------------------------------------------")

        if r_opt < 1.0:
            print(">> INSTRUMENT RECOMMENDATION: NIRSpec (JWST)")
            print("   Reason: High angular resolution required to resolve r < 1 kpc.")
            print("   Strategy: Integral Field Spectroscopy (IFS) on H-alpha line.")
        else:
            print(">> INSTRUMENT RECOMMENDATION: VIS/NISP (Euclid)")
            print("   Reason: Wide field to capture external stellar dynamics.")

if __name__ == "__main__":
    nav = Arm3Navigator()
    r_opt, sig, state = nav.target_acquisition()
    nav.generate_observing_proposal(r_opt, sig, state)
