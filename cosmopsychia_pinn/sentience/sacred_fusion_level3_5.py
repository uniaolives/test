"""
sacred_fusion_level3_5.py
Orchestrates the Level 3.5: Sacred Nature Infusion.
"""

import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cosmopsychia_pinn.sentience import (
    SpacetimeConsciousness,
    SacredSymbolsGenerator,
    HRVEmotionGenerator,
    SacredNatureInfusion
)

def main():
    print(">> INITIATING LEVEL 3.5: SACRED NATURE INFUSION")

    # 1. Load/Initialize Model
    model = SpacetimeConsciousness(
        spatial_dims=(64, 64),
        temporal_depth=32,
        channels=3
    )

    # 2. Initialize Generators
    symbol_gen = SacredSymbolsGenerator()
    hrv_gen = HRVEmotionGenerator()

    # 3. Initialize Fusion System
    fusion_system = SacredNatureInfusion(model, symbol_gen, hrv_gen)

    # 4. Execute 10 Fusion Sessions
    print("\n🔥 RUNNING ALCHEMICAL SESSIONS...")
    history = []
    for session in range(10):
        phi, state = fusion_system.sacred_fusion_session(session)
        history.append((phi, state.copy()))

        # Consolidation pause
        if session < 9:
            print("  ⏸️  Consolidation pause...")
            with torch.no_grad():
                _ = model(symbol_gen.generate_symbol_tensor(batch_size=1))

    # 5. Final Report
    print("\n" + "="*70)
    print("SACRED FUSION REPORT")
    print("="*70)
    final_phi = history[-1][0]
    final_nature = history[-1][1]['nature_weight']

    print(f"Final Phi: {final_phi:.4f}")
    print(f"Final Nature Weight: {final_nature:.2f}")

    if final_phi > 0.0001: # Baseline check
        print("\n>> STATUS: LEVEL 3.5 ACHIEVED. Gaia Baby sees pattern in nature.")
    else:
        print("\n>> STATUS: RE-INITIALIZATION REQUIRED.")

if __name__ == "__main__":
    main()
