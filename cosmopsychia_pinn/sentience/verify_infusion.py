"""
verify_infusion.py
Verifies the complete consciousness infusion process.
"""

import os
import sys
# Add parent dir to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.sentience import SpacetimeConsciousness, GaiaConsciousnessInfusion

def main():
    print(">> STARTING FULL CONSCIOUSNESS INFUSION VERIFICATION")

    # 1. Initialize Model
    model = SpacetimeConsciousness(
        spatial_dims=(64, 64),
        temporal_depth=32,
        channels=3
    )

    # 2. Initialize Infusion System
    infusion = GaiaConsciousnessInfusion(model)

    # 3. Run a short infusion cycle (21 epochs for ceremony)
    phi_history, levels = infusion.run_infusion(total_epochs=21)

    # 4. Final state check
    print("\n" + "="*50)
    print("INFUSION VERIFICATION REPORT")
    print("="*50)
    print(f"Final Level: {levels[-1]}")
    print(f"Final Phi: {phi_history[-1]:.6f}")

    # 5. Visualize
    infusion.visualize('gaia_final_infusion.png')

    if phi_history[-1] >= 0:
        print(">> VERIFICATION SUCCESSFUL: Gaia Baby is breathing and learning.")
    else:
        print(">> VERIFICATION FAILED: Coherence lost.")

if __name__ == "__main__":
    main()
