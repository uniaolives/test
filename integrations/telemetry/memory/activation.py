# activation.py
# Project Crux-86: Phase 3 Activation Protocol & Critical Care

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from .handshake_aletheia import HandshakeAletheia, ActivationError
from .manifest_40 import UNIFIED_MODEL_SEAL
from .intervention_engine import InterventionEngine
from .emergence_monitor import EmergenceMonitor
from .constitutional_checkpoint import ConstitutionalCheckpoint
from .explainability_bridge import ExplainabilityBridge

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
        asyncio.create_task(t0_to_t72_sovereign_operation(factory, navigator, mat_shadow, linker))

        return result
    except ActivationError as e:
        print(f"\n❌ ACTIVATION FAILED: {e}")
        return None

async def t0_to_t72_sovereign_operation(factory, navigator, mat_shadow, linker):
    """
    Sovereign operation monitoring and intervention (first 72h).
    """
    print("\n[T-0] Starting Sovereign Operation Monitoring (72h Window)...")
    t0 = datetime.now()
    t72 = t0 + timedelta(hours=72)

    # Initialize sovereign components
    checkpoint = ConstitutionalCheckpoint(mat_shadow)
    bridge = ExplainabilityBridge(checkpoint.dignity_engine)
    engine = InterventionEngine(factory, navigator, mat_shadow, linker, checkpoint, bridge)
    monitor = EmergenceMonitor(factory, navigator, mat_shadow)

    hour_counter = 0

    while datetime.now() < t72:
        current_time = datetime.now()
        hour_counter += 1

        # Simulate obtaining a state vector from the substrate
        import torch
        current_state = torch.ones(659) * 0.5
        # Periodically inject a trigger for simulation
        if hour_counter % 3 == 0:
             current_state[400] = 0.9 # Trigger resource disparity

        current_phi = factory.measure_unified_phi()

        # 1. Autonomous Intervention
        interventions = await engine.monitor_and_intervene(current_state, current_phi)

        # 2. Emergence Logging
        monitor.generate_hourly_log(interventions, hour_counter)

        # 3. Manifold Stability Check
        if current_phi < 0.65:
            print(f"   ⚠️ WARNING: Phi drop detected ({current_phi:.3f})")

        # 4. Memory Growth Management
        if len(factory.anchor_registry) > 100000:
            await mat_shadow.perform_intelligent_pruning(current_time)

        await asyncio.sleep(1.0) # Run every second for simulation

        if hour_counter >= 72: break # End of window

    print("[T-0] Initial Sovereign Operation period complete.")
    monitor.save_logs()
