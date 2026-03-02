import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add core/python to sys.path to import arkhe
sys.path.append(os.path.join(os.getcwd(), 'core', 'python'))

from arkhe.consensus.quantum_ising import Validador, IsingConsensusEngine
from arkhe.crypto.pq_arkhe import QuantumSafeChannel, ArkheSecureNetwork
from arkhe.desci.protocol import DeSciManifold, Experiment

def test_consensus_engine():
    validadores = [
        Validador("v1", 100.0, 0.9),
        Validador("v2", 150.0, 0.8),
        Validador("v3", 50.0, 0.95),
        Validador("v4", 200.0, 0.7)
    ]
    engine = IsingConsensusEngine(validadores)

    proposta = b"test_transaction_data"
    success, metadata = engine.alcancar_consensus(proposta, max_iter=100)

    assert isinstance(success, bool)
    assert 'decisao' in metadata
    assert 'magnetizacao' in metadata or 'magnetizacao_final' in metadata

def test_quantum_safe_channel():
    node_a = "alice"
    node_b = "bob"
    channel_a = QuantumSafeChannel(node_a, node_b)
    channel_b = QuantumSafeChannel(node_b, node_a)

    # Handshake simulation
    init = channel_a.initiate_handshake()
    resp = channel_b.respond_handshake(init)
    success = channel_a.complete_handshake(resp)

    assert success is True
    assert channel_a.is_established is True

    # Message exchange
    plaintext = b"Hello, Bob!"
    msg = channel_a.send_message(plaintext)

    received_plaintext, anomaly_report = channel_b.receive_message(msg)

    assert received_plaintext == plaintext
    assert anomaly_report['status'] in ['NORMAL', 'INSUFFICIENT_DATA']

def test_desci_manifold():
    manifold = DeSciManifold()

    exp = Experiment(
        experiment_id="",
        title="Test Experiment",
        authors=["Author A"],
        institution="University X",
        hypothesis="Testing if 1+1=2",
        methodology_hash="hash123",
        primary_endpoint="result",
        statistical_plan={"planned_n": 100, "power": 0.9},
        pre_registered_at=datetime.now()
    )

    exp_id = manifold.pre_register_experiment(exp)
    assert exp_id is not None
    assert exp_id in manifold.experiments

    results = {
        "primary_endpoint_result": 2.0,
        "actual_sample_size": 100,
        "analysis_method": None, # Will cause compliance score to drop if expected
        "tested_hypothesis": "Testing if 1+1=2"
    }

    compliance = manifold.submit_results(
        exp_id,
        data_hash="data_hash_456",
        results=results,
        execution_logs=[{"op": "run"}]
    )

    assert 0.0 <= compliance <= 1.0

    phi_data = manifold.calculate_phi_score(exp_id)
    assert 'phi_score' in phi_data
    assert 0.0 <= phi_data['phi_score'] <= 1.0
