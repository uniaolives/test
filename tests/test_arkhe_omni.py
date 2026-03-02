import pytest
import asyncio
import numpy as np
from arkhe_omni_kernel import ArkheOmniKernel, AttentionNode, MemeticPacket, CoherenceVector, PHI

@pytest.mark.asyncio
async def test_kernel_pulse():
    kernel = ArkheOmniKernel()
    # Run for a few cycles
    task = asyncio.create_task(kernel.gamma_cycle())
    await asyncio.sleep(0.1) # Should cover ~4 cycles
    kernel.running = False
    await task
    assert kernel.safe_core.coherence > 0.8
    assert kernel.network.global_phi >= 0.0

def test_node_assimilation():
    node = AttentionNode("TestNode")
    packet = MemeticPacket(source_id="Source", content="Test Insight", phi=1.0,
                          context=CoherenceVector(np.random.randn(128)).normalize())

    # Force high resonance
    node.phi = 0.5
    packet.phi = 1.0
    # Match spins
    packet.spin = node.spin

    assimilated = node.receive(packet)
    assert assimilated is True
    assert packet.id in node.memory
    assert node.phi > 0.5

def test_network_coherence():
    kernel = ArkheOmniKernel()
    initial_phi = kernel.network.global_coherence()

    # Artificially increase node phis
    for node in kernel.network.nodes.values():
        node.phi = 1.0

    new_phi = kernel.network.global_coherence()
    assert new_phi == 1.0
    assert new_phi > initial_phi

def test_safe_core_veto():
    from arkhe_omni_kernel import SafeCore
    safe = SafeCore()
    # PHI_CRITICAL is 0.847. F = 1 - C.
    # To drop C below 0.847, F must be > 0.153
    assert safe.regulate(0.1) is True
    assert safe.regulate(0.1) is False # Total F = 0.2, C = 0.8 < 0.847
