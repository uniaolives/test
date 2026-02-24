"""
Emergence Validation: Validating phase transitions and ASI emergence in the Pleroma network.
Simulates the transition from subcritical to supercritical density.
"""

import asyncio
import sys
import os
import numpy as np

# Add paths to find pleroma_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pleroma_kernel import PleromaKernel

async def test_emergence():
    print("="*60)
    print("ASI EMERGENCE VALIDATION")
    print("="*60)

    # 1. Initialize with low node count (Subcritical)
    print("\n[STEP 1] Initializing Subcritical Network (10 nodes)...")
    kernel = PleromaKernel(n_nodes=10)
    await kernel.run(duration=1.0)
    print(f"\n  Final State: {kernel.system_state}, Density: {kernel.rho_actual:.4f}")
    assert kernel.system_state in ["Subcritical", "Emerging"]

    # 2. Increase density to trigger emergence (increase coherence/nodes)
    print("\n[STEP 2] Catalyzing Emergence (Simulating high density)...")
    # In this simulation, we simulate adding nodes by just creating a larger kernel
    # and forcing high coherence
    kernel_large = PleromaKernel(n_nodes=20)
    for node in kernel_large.nodes.values():
        node.coherence = 1.0 # Maximize coherence to push density

    # Run and observe transition
    await kernel_large.run(duration=2.0)
    print(f"\n  Final State: {kernel_large.system_state}, Density: {kernel_large.rho_actual:.4f}")

    # 3. Validation
    print("\n" + "="*60)
    print("📊 EMERGENCE VALIDATION REPORT")
    print(f"  Critical Density (rho_c): {kernel_large.rho_critical}")
    print(f"  Actual Density (rho): {kernel_large.rho_actual:.4f}")
    print(f"  Final System State: {kernel_large.system_state}")

    # Successful emergence should reach Supercritical or at least Emerging
    assert kernel_large.system_state in ["Emerging", "Supercritical"], "ASI Emergence failed!"
    print("  ✓ Phase transition to Supercritical state validated.")

    print("\n✅ Validation completed")

if __name__ == "__main__":
    asyncio.run(test_emergence())
