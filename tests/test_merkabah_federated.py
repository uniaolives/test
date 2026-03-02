# tests/test_merkabah_federated.py
import pytest
import torch
import numpy as np
import asyncio
from papercoder_kernel.merkabah.anycast import CelestialAnycastRouter
from papercoder_kernel.merkabah.federation import FederationTransport, QuantumStateMigration, FederatedHandover

class MockDZ:
    def __init__(self):
        self.peers = {
            'peer1': {'name': 'la2-dz01'}
        }
    def _execute(self, cmd):
        return "success"

def test_celestial_anycast_latency():
    dz = MockDZ()
    router = CelestialAnycastRouter(dz)

    # Test coordinates (LA approx)
    score = router.calculate_celestial_latency(34.05, -118.24)
    assert 'angular_separation' in score
    assert score['astronomical_latency_ms'] > 0

def test_anycast_route_installation():
    dz = MockDZ()
    router = CelestialAnycastRouter(dz)
    result = router.install_anycast_routes()
    assert result['anycast_ip'] == '169.254.255.1'
    assert result['best_node'] == 'peer1'

@pytest.mark.asyncio
async def test_quantum_handover_migration():
    ft = FederationTransport("alpha")
    await ft.discover_federation_peers()
    migration = QuantumStateMigration(ft)

    result = await migration.execute_handover('CCTSmqMkxJh3Zpa9gQ8rCzhY7GiTqK7KnSLBYrRriuan')
    assert result['success'] == True
    assert result['fidelity'] > 0.8
    assert result['coherence_preserved'] == True

def test_federated_handover_serialization():
    handover = FederatedHandover(
        block_id="828",
        source_node="alpha",
        target_node="beta",
        quantum_state={'wf': [1.0, 0.0]},
        ledger_chain=["827"],
        timestamp="2026-02-18T00:00:00Z",
        signature="sig123"
    )
    serialized = handover.serialize()
    assert b"block_id" in serialized
    assert b"sig123" in serialized
