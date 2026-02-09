import pytest
import numpy as np
import asyncio
from src.avalon.pop.circuits.oracle import PersistentOrderOracle
from src.avalon.pop.components.node import OperationalQCN, SpectralCube
from src.avalon.pop.integration.merkabah import MERKABAHPOPAdapter

def test_oracle_detection_logic():
    oracle = PersistentOrderOracle()

    # Case 1: Likely Life
    res = oracle.simulate_detection(0.9, 0.8, 0.85)
    assert res['is_life_detected'] is True
    assert res['detection_probability'] > 0.8

    # Case 2: Unlikely Life (Noise)
    res = oracle.simulate_detection(0.1, 0.2, 0.1)
    assert res['is_life_detected'] is False

def test_qcn_feature_extraction():
    node = OperationalQCN(node_id="test-node")

    # Create cube with clear oscillation (DNE)
    data = np.zeros((10, 10, 8, 5))
    for t in range(5):
        data[:,:,:,t] = np.sin(2 * np.pi * t / 5) * 5.0

    cube = SpectralCube(data=data, coordinates={"x": 0, "y": 0, "z": 0})
    features = node.extract_features(cube)

    assert features['dne'] > 0.8 # Should be high due to sine wave

@pytest.mark.asyncio
async def test_qcn_evaluation_flow():
    node = OperationalQCN(node_id="test-node")

    # High probability cube
    data = np.random.randn(10, 10, 8, 5)
    for t in range(5):
        data[:,:,:,t] += np.sin(2 * np.pi * t / 5) * 10.0

    cube = SpectralCube(data=data, coordinates={"x": 0, "y": 0, "z": 0})
    res = await node.evaluate_cube(cube)

    assert res['is_life_detected'] is True
    assert node.protocol_state in ["CURIOSITY", "DISCOVERY"]

@pytest.mark.asyncio
async def test_merkabah_adapter():
    node = OperationalQCN(node_id="test-node")
    adapter = MERKABAHPOPAdapter(node, "http://mock-merkabah")

    command = {
        "id": "cmd-001",
        "command": "SCAN_REGION",
        "parameters": {
            "coordinates": {"x": 10, "y": 20},
            "simulate_life": True
        }
    }

    response = await adapter.handle_command(command)
    assert response['status'] == "COMPLETED"
    assert response['result']['is_life_detected'] is True

    # Test Query State
    command_query = {"id": "cmd-002", "command": "QUERY_PO_STATE", "parameters": {}}
    response_query = await adapter.handle_command(command_query)
    assert response_query['result']['node_id'] == "test-node"
