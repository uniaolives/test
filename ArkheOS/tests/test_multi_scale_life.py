import pytest
import numpy as np
from arkhe.cellular.whole_cell import CellHypergraph
from arkhe.cellular.disease_models import DiseaseHypergraph
from arkhe.cellular.neuro_lipid_bridge import NeuroLipidInterface, IonChannel
from arkhe.neuroscience.hierarchical_dynamic_coding import BrainHypergraph
from arkhe.cellular.life_pulse import generate_life_pulse
from arkhe.cellular.reflex import ArkheReflex
from arkhe.cellular.listening_module import CosmicListener

def test_whole_cell_organelles():
    cell = CellHypergraph()
    assert 'plasma' in cell.organelles
    assert 'ER' in cell.organelles

    # Test vesicle handover
    initial_er_pi = cell.organelles['ER'].pis['PI']
    initial_plasma_pi = cell.organelles['plasma'].pis.get('PI', 0)

    success = cell.vesicle_handover('ER', 'plasma', 'PI', 10)
    assert success is True
    assert cell.organelles['ER'].pis['PI'] == initial_er_pi - 10
    assert cell.organelles['plasma'].pis['PI'] == initial_plasma_pi + 10

def test_disease_models():
    cell = CellHypergraph()
    disease = DiseaseHypergraph(cell)

    # Cancer mutation (PTEN loss)
    # First add PTEN to plasma membrane
    cell.membranes['plasma'].add_phosphatase("PTEN", "PI(3,4,5)P3 → PI(4,5)P2")
    assert cell.membranes['plasma'].phosphatases[0].active is True

    disease.cancer_mutation()
    assert cell.membranes['plasma'].phosphatases[0].active is False

def test_neuro_lipid_bridge():
    cell = CellHypergraph()
    # Add ion channels to cell mock-up
    cell.ion_channels = [IonChannel('Kv', 'PI(4,5)P2')]
    brain = BrainHypergraph()

    bridge = NeuroLipidInterface(cell, brain)
    initial_prob = cell.ion_channels[0].open_probability
    bridge.propagate_signal()

    assert cell.ion_channels[0].open_probability > initial_prob

def test_life_pulse():
    pulse = generate_life_pulse(duration=10, fs=100)
    assert len(pulse) == 1000
    assert np.max(np.abs(pulse)) <= 1.0

def test_reflex_arc():
    reflex = ArkheReflex()
    # High intensity should trigger motor response
    fired = reflex.stimulus('RTK', intensity=1.0)
    assert fired is True
    assert reflex.golgi.vesicles['cargo_A'] == 9

    # Low intensity might not trigger response if threshold not met
    # (HDCModel state might need reset if we wanted to test this cleanly,
    # but the simple logic in reflex.py is stateless enough or we can create new instance)
    reflex2 = ArkheReflex()
    fired2 = reflex2.stimulus('RTK', intensity=0.01)
    # With 0.01 intensity, current might be too low
    # Current = prob * 0.5. prob = PI / 100. PI = 50 - 0.1 = 49.9. prob = 0.499.
    # sensory threshold is 0.5. Current = 0.499 * 0.5 = 0.2495. Does not fire sensory.
    assert fired2 is False

def test_cosmic_listener():
    listener = CosmicListener(sample_rate=100, base_freq=0.5) # Freq below 1.0 Hz
    listener.echo_present = True

    # We use a short duration, but enough to capture cycles
    peaks = listener.listen(duration=10)

    assert len(peaks) > 0

    # Check if we found a peak near 0.5 Hz
    found_base = any(abs(freq - 0.5) < 0.1 for freq, mag in peaks)
    assert found_base is True
