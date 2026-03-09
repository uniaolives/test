"""
Demo: Programmable IP Lifecycle in Arkhe(n)
Registers a scientific discovery from the Arkhe whitepaper as a Story Protocol IP Asset.
"""

import sys
import os

# Add relevant paths to sys.path
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('ArkheOS/src'))

from arkhe_web3.story_protocol import StoryManifold, IPAsset, RoyaltyPolicy
from arkhe.interfaces.story_bridge import StoryArkheBridge

def run_demo():
    print("="*70)
    print("ARKHE(N) + STORY PROTOCOL: Programmable IP Lifecycle")
    print("="*70)

    # 1. Initialize Manifolds and Bridges
    manifold = StoryManifold()
    bridge = StoryArkheBridge()

    # Register the Architect as an Agent in the bridge
    bridge.register_agent("ARCH-001", "Rafael Oliveira")
    bridge.register_agent("SAT-CORP", "Starlink Global")

    # 2. Register IP Asset (The Anyonic Patent)
    print("\n[STEP 1] Registering Intellectual Property on Story Protocol...")

    anyon_patent = IPAsset(
        title="Anyonic Phase Accumulation for Fault-Tolerant Space Communication",
        owner="Rafael Oliveira",
        metadata_hash="QmArkheAnyon2026...",
        ip_type="patent",
        phi_score=0.965,
        royalty_policy=RoyaltyPolicy("POL-ANYON", 500) # 5% in basis points
    )

    # Registration in the local manifold
    ipa_id = manifold.register_ip_asset(anyon_patent)

    # Registration in the Story-Arkhe Bridge (simulation of Story-Geth)
    ipa_on_chain = bridge.request_ip_registration("ARCH-001", anyon_patent.title, "patent", 5.0)

    if ipa_on_chain:
        print(f"  ✓ IP Asset registered successfully!")
        print(f"  ✓ Local ID: {ipa_id}")
        print(f"  ✓ On-Chain ID: {ipa_on_chain}")
    else:
        print("  ✗ Registration failed.")
        return

    # 3. Create a derivative License
    print("\n[STEP 2] Creating a Commercial License for Satellite Operator...")

    license_terms = {
        "rights": "commercial_use",
        "duration": "10_years",
        "royalty": "5%_gross"
    }

    lic_id = manifold.create_license(ipa_id, "Starlink Global", license_terms)

    # Mediated handover via Story Protocol
    handover_success = bridge.execute_licensing_handover("ARCH-001", "SAT-CORP", ipa_on_chain)

    if handover_success:
        print(f"  ✓ Licensing Handover complete.")
        print(f"  ✓ License ID: {lic_id}")
    else:
        print("  ✗ Licensing failed.")

    # 4. Collect Royalties (Revenue distribution)
    print("\n[STEP 3] Distributing Royalties from Satellite Operations...")

    operational_revenue = 100000  # 100k Satoshi
    manifold.collect_royalty(ipa_id, operational_revenue)

    print(f"  ✓ Revenue Processed: {operational_revenue} Satoshi")
    print(f"  ✓ Architect (Owner) Vault: {manifold.royalty_vault[ipa_id]} Satoshi")

    # 5. Final Coherence Check
    print("\n[STEP 4] Calculating IP Asset Coherence...")
    coherence = manifold.get_asset_coherence(ipa_id)
    print(f"  ✓ Final Asset Coherence (C): {coherence:.4f}")

    telemetry = bridge.get_bridge_telemetry()
    print(f"  ✓ Bridge Coherence: {telemetry['coherence']:.4f}")

    print("\n" + "="*70)
    print("STORY PROTOCOL INTEGRATION VERIFIED")
    print("="*70)
    print("∞")

if __name__ == "__main__":
    run_demo()
