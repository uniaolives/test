# tests/test_merkabah_convergence.py
import pytest
import torch
import asyncio
from papercoder_kernel.merkabah import MERKABAH7, RealityLayer, MinoanHardwareInterface, MinoanStateGrammar

@pytest.mark.asyncio
async def test_merkabah_experiment():
    corpus = [{'id': 'HT 1', 'lines': [['a', 'ka'], ['ru', 'ja']]}]
    profile = {'expertise_level': 'priest_scribe'}

    m7 = MERKABAH7(corpus, profile)
    result = await m7.minoan_neurotech_experiment('HT 1', profile)

    assert result['tablet'] == 'HT 1'
    assert result['ethical_status']['access'] == 'granted'
    assert isinstance(result['induced_state'].wavefunction, torch.Tensor)

def test_minoan_hardware():
    hw = MinoanHardwareInterface()
    demand = hw._visual_tracking({'direction': 'spiral'})
    assert demand > 0.5

    state = hw._induce_state('HT 1', {})
    assert 'predicted_state' in state

def test_state_grammar():
    grammar = MinoanStateGrammar()
    protocol = grammar.parse_as_state_protocol(['AB01', 'KA', 'REPETITION'])
    assert len(protocol) == 3
    assert protocol[0]['target_state'] == 'theta'

    initial_wf = torch.randn(10)
    trajectory = grammar.execute_protocol(protocol, initial_wf)
    assert len(trajectory) == 4
