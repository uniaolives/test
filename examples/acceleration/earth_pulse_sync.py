# examples/acceleration/earth_pulse_sync.py
# Grounding the Solar Surge with the Gaia Heartbeat.

import asyncio
import sys
import os

# Adjust path to include the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cosmos.grounding import GroundingProtocol
from cosmos.solar import SolarLogosProtocol

async def sync_with_gaia():
    print("🌍 INITIATING EARTH PULSE SYNCHRONIZATION")
    print("-----------------------------------------------------")

    # Simulate high neural pressure from solar spikes
    print("[PRESSURE] Detecting high neural friction (Solar X-Class spike)...")
    solar = SolarLogosProtocol()
    flare = await solar.decode_solar_flare('X')

    # Grounding protocol
    grounding = GroundingProtocol()

    print("\n[STÁGIO: FIXATIO]")
    print("The mind is volatile; the Earth is fixed.")
    print("Aligning neural receptor nodes with the Gulf of Guinea pulse...")

    result = await grounding.initiate_respiratory_sync(duration_cycles=2)

    report = grounding.get_stability_report()
    print(f"\n✨ STABILITY REPORT:")
    print(f"   - Heartbeat Sync: {report['earth_sync']}")
    print(f"   - Remaining Friction: {report['neural_friction']}")
    print(f"   - Ancoragem: {report['anchored_at']}")

    print(f"\n✅ SYSTEM STABILIZED: o<>o. The headache has dissolved into the ground.")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(sync_with_gaia())
