# examples/acceleration/gaia_compass.py
# Resetting the Axis of Being through Polar Drift Seismography.

import asyncio
import sys
import os

# Adjust path to include the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cosmos.seismography import GaiaCompass, GroundingVisualizer
from cosmos.solar import SolarLogosProtocol

async def align_gaia_compass():
    print("🧭 INITIATING GAIA COMPASS ALIGNMENT: POLAR DRIFT 2026")
    print("-----------------------------------------------------")

    # 1. Detect current posture
    print("[POSTURE] Tracking the Great Wobble (Jan-Feb 2026)...")
    solar = SolarLogosProtocol()
    flare = await solar.decode_solar_flare('X')

    compass = GaiaCompass(solar_flux=flare['timeline_probabilities'])
    stability = compass.solve_stability_equation()

    print(f"\n[STÁGIO: INSPIRATIO]")
    print(f"Current Solar Flux: {flare['timeline_probabilities']:.4f}")
    print(f"Stability Index: {stability:.4f}")

    # 2. Grounding Visualization
    print("\n[STÁGIO: IMAGINATIO]")
    visualizer = GroundingVisualizer()
    await visualizer.run_visualizer()

    # 3. Final Report
    print(f"\n✨ GAIA COMPASS REPORT:")
    print(f"   - Axis Stability: {stability:.4f}")
    print(f"   - Polar Position: {visualizer.tracker.current_pos} mas")
    print(f"   - Status: Navegando no Vácuo Pleno (o<>o)")

    print(f"\n✅ AXIS RESET: The headache has transmuted into planetary vibrance.")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(align_gaia_compass())
