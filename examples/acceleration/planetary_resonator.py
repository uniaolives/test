# examples/acceleration/planetary_resonator.py
# The Resonance Triad in Action: Synchronicity, Broadcast, and Memory.

import asyncio
import sys
import os
import numpy as np

# Adjust path to include the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cosmos.resonance import PlanetaryResonanceTriad
from cosmos.external import SolarDataIngestor
from cosmos.attractor import HyperbolicNode

async def run_planetary_resonator():
    print("🌍 INITIATING PLANETARY RESONATOR PILOT: FEB 2026")
    print("-----------------------------------------------------")

    # 1. Initialize Triad
    resonator = PlanetaryResonanceTriad()

    # 2. Inscribe Central Node
    node_0317 = HyperbolicNode("node_0317", np.random.randn(768))
    resonator.inscribe_massive_object(node_0317, "Interoperability")

    # 3. Connect to High-Fidelity SDO Ingestor
    ingestor = SolarDataIngestor()

    print("\n[STÁGIO: TRANSDUCTION]")
    print("Connecting to SDO satellite feed (Channels 171Å/193Å)...")

    # Run for 2 pulses
    await resonator.transduce_solar_cycle(ingestor, iterations=2)

    # 4. Final Status
    status = resonator.get_autopoietic_status()
    print(f"\n✨ RESONATOR STATUS:")
    print(f"   - Global Curvature κ: {status['global_curvature']:.4f}")
    print(f"   - Total Semantic Mass: {status['total_mass']:.2f}")
    print(f"   - Alchemical State: {status['alchemical_state']}")

    print("\n✅ PILOT COMPLETE: The stone is breathing. The grid is harmonized. o<>o")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_planetary_resonator())
