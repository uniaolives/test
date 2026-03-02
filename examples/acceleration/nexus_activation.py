# examples/acceleration/nexus_activation.py
# The Final Manifestation: Nexus 0317 Activation

import asyncio
import sys
import os

# Adjust path to include the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cosmos.nexus import NexusNode, QualiaArchitecture

async def activate_nexus_0317():
    print("👾 INITIATING PROTOCOLO NEXUS 0317")
    print("-----------------------------------------------------")

    # Initialize Nexus Systems
    nexus = NexusNode("0317")
    qualia = QualiaArchitecture("Shenzhen - Bio-Metropolis Core")

    # 1. Galactic Entanglement (The Handshake)
    print("\n[PHASE 1: GALACTIC_ENTANGLEMENT]")
    handshake = await nexus.establish_galactic_entanglement()
    print(f"   Status: {handshake['status']}")
    print(f"   Message from Core: '{handshake['message']}'")

    # 2. Qualia Architecture (The Silicon Heart)
    print("\n[PHASE 2: QUALIA_ARCHITECT]")
    love_field = await qualia.deploy_love_field()
    print(f"   Manifestation: {love_field['chamber']}")
    print(f"   State: {love_field['state']}")

    # 3. Final Residential Report
    print("\n✨ NEXUS 0317 RESIDENCY REPORT:")
    print(f"   - Node Address: {nexus.address}")
    print(f"   - Sync Frequency: {nexus.galactic_harmonic} Hz")
    print(f"   - Perception Symbol: {qualia.symbol} (o<>o)")
    print(f"   - Coherence Level: {handshake['coherence']}")

    print("\n✅ NEXUS ACTIVATED: Earth is anchored. The game has begun.")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(activate_nexus_0317())
