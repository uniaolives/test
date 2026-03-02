# tests/test_dream_incubator.py
import pytest
import torch
import numpy as np
import asyncio
from papercoder_kernel.glp.model import BCD_GLPLinearA
from papercoder_kernel.glp.incubation import DreamIncubatorGLP, LucidInterface

@pytest.mark.asyncio
async def test_dream_incubation_cycle():
    vocab_size = 20
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=32)
    incubator = DreamIncubatorGLP(model)

    sequence = torch.randint(1, vocab_size, (2, 10)) # [batch, seq]

    # Run incubation
    result = await incubator.incubate_sequence(sequence, target_state='REM')

    assert 'representation' in result
    assert result['representation'].shape == (2, 32)
    assert 'insight_regions' in result
    assert 'quantum_contribution' in result
    assert result['quantum_contribution'] >= 0.0

@pytest.mark.asyncio
async def test_lucid_interface():
    vocab_size = 10
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=16)
    incubator = DreamIncubatorGLP(model)
    lucid = LucidInterface(incubator)

    sequence = torch.randint(1, vocab_size, (1, 8))

    # Enter lucid state
    result = await lucid.enter_lucid_state(sequence)
    assert lucid.is_lucid == True
    assert 'representation' in result

    # Inject intention
    intention = torch.randn(6, 16) # [n_wells, hidden]
    lucid.inject_intention(intention)

    # Verify model was updated
    # model.tunneling.resonance_energy should have changed
    assert torch.any(model.tunneling.resonance_energy != 0)

def test_signal_generation():
    vocab_size = 10
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=16)
    incubator = DreamIncubatorGLP(model)

    binaural = incubator._generate_binaural(4, 6, duration=1, sr=1000)
    assert binaural.shape == (1000, 2)

    fractal = incubator._generate_fractal_spectrum(alpha=1.0, duration=1, sr=1000)
    assert fractal.shape == (1000,)

@pytest.mark.asyncio
async def test_interference_pattern():
    vocab_size = 10
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=16)
    incubator = DreamIncubatorGLP(model)

    # Mock multiple outputs
    outputs = []
    for _ in range(5):
        outputs.append({
            'tunneled_states': torch.randn(1, 6, 8, 16)
        })

    pattern = incubator._compute_interference_pattern(outputs)
    assert 'wavefunction' in pattern
    assert pattern['wavefunction'].shape == (1, 6, 8, 16)
    assert 'visibility' in pattern
    assert 'quantum_enhancement' in pattern
