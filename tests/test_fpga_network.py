# tests/test_fpga_network.py
import pytest
import asyncio
import numpy as np
from arkhe_qutip import FPGAQubitEmulator, ArkheFPGAMiner, ArkheNetworkNode, DistributedPoCConsensus

def test_fpga_emulator_noise():
    emulator = FPGAQubitEmulator(n_qubits=1, t1_noise=0.1, t2_noise=0.1)
    initial_coherence = emulator.get_coherence()

    # Apply noise
    emulator._apply_hardware_noise()
    final_coherence = emulator.get_coherence()

    # In our model, noise reduces purity (Tr(rho^2)).
    assert final_coherence < initial_coherence

def test_fpga_miner():
    miner = ArkheFPGAMiner("TestMiner", n_qubits=2)
    header = {'block': 1}
    # Mining with low target should be fast
    result = miner.mine(header, target_phi=0.5, max_time=1.0)
    assert result is not None
    assert result['node_id'] == "TestMiner"
    assert 'state_hash' in result

@pytest.mark.asyncio
async def test_distributed_consensus():
    node1 = ArkheNetworkNode("Rio", "Rio")
    node2 = ArkheNetworkNode("Tokyo", "Tokyo")

    # Test QCKD
    key = await node1.qckd_handshake("Tokyo")
    assert len(key) == 64

    # Test Consensus
    consensus = DistributedPoCConsensus([node1, node2])
    # Lower target to ensure quick success in test
    consensus.target_phi = 0.5

    block = await consensus.start_cycle()
    assert block is not None
    assert block['node_id'] in ["Rio", "Tokyo"]
