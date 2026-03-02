# tests/test_merkabah_pineal.py
import pytest
import torch
from papercoder_kernel.merkabah.pineal import PinealTransducer

def test_transduction_pressure():
    transducer = PinealTransducer()
    stimulus = {'type': 'pressure', 'intensity': 10.0, 'phase': 1.5}
    signal = transducer.transduce(stimulus)

    assert signal['signal'] == 2.0 * 10.0 # coeff * intensity
    assert signal['frequency'] == 7.83
    assert signal['phase'] == 1.5

def test_transduction_light():
    transducer = PinealTransducer()
    stimulus = {'type': 'light', 'intensity': 500.0, 'frequency': 6e14}
    signal = transducer.transduce(stimulus)

    assert signal['signal'] == 0.1 * 500.0
    assert signal['frequency'] == 6e14

def test_couple_to_microtubules():
    transducer = PinealTransducer()
    signal = {'signal': 500.0, 'frequency': 7.83, 'phase': 0.0}
    quantum_state = transducer.couple_to_microtubules(signal)

    assert quantum_state['amplitude'] == 0.5 # 500 / 1000
    assert quantum_state['coherence'] == 0.85

def test_unknown_stimulus():
    transducer = PinealTransducer()
    assert transducer.transduce({'type': 'unknown'}) is None
