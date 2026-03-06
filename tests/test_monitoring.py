import pytest
import asyncio
from gateway.app.monitoring.listener import RealityListener, QualicPulse
from datetime import datetime

@pytest.mark.asyncio
async def test_reality_listener_logic():
    listener = RealityListener()
    # Test pulse acquisition
    q_pulse = await listener.acquire_quantum_pulse()
    assert q_pulse.source == 'quantum_sim'
    assert 0.0 <= q_pulse.q_value <= 1.0

    # Test modulation
    initial_q = listener.current_q
    listener.modulate(q_pulse)
    assert listener.current_q != initial_q
    assert len(listener.coherence_history) == 1

def test_reality_monitor_integration():
    listener = RealityListener()
    # Manual update of monitor via listener-like logic
    listener.monitor.update_reality_index('DONE', 2.0)
    stats = listener.get_state()
    assert stats['s_index'] > 0
    assert stats['state'] != "INICIAL"
