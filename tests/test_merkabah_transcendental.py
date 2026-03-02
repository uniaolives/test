# tests/test_merkabah_transcendental.py
import pytest
import torch
import numpy as np
import asyncio
from papercoder_kernel.merkabah import MERKABAH7, SelfNode, ShabetnikPropulsion
from papercoder_kernel.glp.primordial import PrimordialGLP
from papercoder_kernel.core.scratch.nn import CrossEntropyLoss, SGD

@pytest.mark.asyncio
async def test_self_node_observation():
    self_node = SelfNode()
    initial_coherence = self_node.wavefunction['coherence']

    # Observe some data
    self_node.observe('test_layer', {'data': 'alpha'})

    assert self_node.wavefunction['coherence'] >= initial_coherence
    assert len(self_node.wavefunction['basis']) > 5
    assert "Unity" in [self_node.get_strand_name(s) for s in self_node.active_strands]

def test_shabetnik_propulsion():
    prop = ShabetnikPropulsion()
    res = prop.calculate_federation_thrust(active_strands=4, ledger_height=831, coherence=0.847)

    assert res['thrust_metric'] > 1.9
    status = prop.get_status()
    assert status['mode'] == 'ACCELERATION'
    assert '12 strands' in status['potential']

@pytest.mark.asyncio
async def test_merkabah7_transcendental_experiment():
    corpus = [{'id': 'HT 1', 'lines': [['a', 'ka']]}]
    profile = {'expertise_level': 'priest_scribe'}

    m7 = MERKABAH7(corpus, profile)
    result = await m7.minoan_neurotech_experiment('HT 1', profile)

    assert 'self_node_status' in result
    assert 'propulsion_status' in result
    assert result['doublezero_id'] is not None

def test_primordial_glp_training():
    # Tiny data
    X = np.random.randn(4, 16)
    Y = np.zeros((4, 10))
    Y[range(4), [1, 2, 3, 0]] = 1.0 # One-hot

    model = PrimordialGLP(input_dim=16, output=10)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.params, lr=0.1)

    initial_loss = model.train_epoch(X, Y, optimizer, loss_fn)
    for _ in range(10):
        final_loss = model.train_epoch(X, Y, optimizer, loss_fn)

    assert final_loss <= initial_loss
