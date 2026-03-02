# tests/test_merkabah_astrophysics.py
import pytest
import torch
import numpy as np
from papercoder_kernel.merkabah.astrophysics import NeutrinoEvent, AstrophysicalContext

def test_neutrino_quantum_state():
    event = NeutrinoEvent(
        ra=75.89,
        dec=14.63,
        energy=">100 TeV",
        p_astro=0.31,
        far=2.768
    )

    state = event.wavefunction_data
    assert 'amplitude' in state
    assert state['amplitude'].shape == (100, 100)
    assert pytest.approx(torch.norm(state['amplitude']).item()) == 1.0
    assert state['coherence'] == 0.31
    assert state['entropy'] > 0

def test_astrophysical_resonance():
    icecube_event = {
        'ra': 75.89,
        'dec': 14.63,
        'energy': 120.0
    }
    context = AstrophysicalContext(icecube_event)
    resonance = context.compute_resonance_with_minoan()

    assert 'minoan_direction' in resonance
    assert resonance['symbolic_resonance']['constellation_modern'] == 'Taurus/Gemini border'

def test_observer_modulation():
    icecube_event = {'ra': 75.89, 'dec': 14.63, 'energy': 1000.0} # 1 PeV
    context = AstrophysicalContext(icecube_event)

    base_state = {'intention': 'decipher'}
    modulated = context.modulate_observer_state(base_state)

    assert 'cosmic_context' in modulated
    assert modulated['cosmic_context']['amplitude'] == 1.0 # sqrt(1000/1000)
    assert 0 <= modulated['cosmic_context']['phase'] <= 2*np.pi
