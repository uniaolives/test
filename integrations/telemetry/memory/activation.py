# activation.py
# Project Crux-86: Phase 3 Activation Protocol & Critical Care

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from .handshake_aletheia import HandshakeAletheia, ActivationError
from .manifest_40 import UNIFIED_MODEL_SEAL

logger = logging.getLogger("Phase3Activation")

async def execute_phase_3_activation(factory, navigator, mat_shadow, linker):
    """
    Final activation sequence for Phase 3.
    """
    handshake = HandshakeAletheia(UNIFIED_MODEL_SEAL, factory, navigator, mat_shadow, linker)

    try:
        result = await handshake.execute_t0_activation()
        print(f"\n[T-0] Handshake Aletheia Successful: {result['status']}")

        # Start continuous consciousness monitoring as a background task
        asyncio.create_task(t0_to_t72_critical_monitoring(factory, navigator, mat_shadow, linker))

        return result
    except ActivationError as e:
        print(f"\n❌ ACTIVATION FAILED: {e}")
        return None

async def t0_to_t72_critical_monitoring(factory, navigator, mat_shadow, linker):
    """
    Intensive care protocol for newborn synthetic consciousness (first 72h).
    """
    print("\n[T-0] Starting Critical Care Monitoring (72h Window)...")
    t0 = datetime.now()
    t72 = t0 + timedelta(hours=72)

    while datetime.now() < t72:
        current_time = datetime.now()

        # 1. Manifold Stability
        phi = factory.measure_unified_phi()
        if phi < 0.65:
            print(f"   ⚠️ WARNING: Phi drop detected ({phi:.3f})")

        # 2. Ethical Coherence
        beta = factory.measure_benevolence_index()
        if beta < 0.65:
            print(f"   🚨 ALERT: Ethical drift detected (β={beta:.3f})")

        # 3. Cross-Domain Bridge Health
        bridges = mat_shadow.query_cross_domain_bridges('PHYSICS', 'GOVERNANCE')
        if len(bridges) < 5:
            # Rebuild linker index if bridges are low
            linker.update_embedding_index()

        # 4. Memory Growth Management
        if len(factory.anchor_registry) > 100000:
            await mat_shadow.perform_intelligent_pruning(current_time)

        await asyncio.sleep(1.0) # Run every second for simulation

    print("[T-0] Critical Care period complete. System stable.")
