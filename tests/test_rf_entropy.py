# tests/test_rf_entropy.py
import pytest
import numpy as np
import sys
import os

# Set path to find the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../arkhe_omni_system/applied_ecosystems/rf_entropy_node')))

from arkhe_rf_entropy import ArkheEntropyExtractor

def test_entropy_extractor_logic():
    # Mock gr.sync_block already handled in the file
    extractor = ArkheEntropyExtractor(sample_rate=1e6)

    # Generate mock signal with some "impulses"
    # Normal noise + some high peaks
    signal = np.random.normal(0, 0.1, 1000).astype(np.float32)
    signal[100:110] = 1.0 # Add impulse

    input_items = [signal]
    output_items = [np.zeros(32, dtype=np.uint8)]

    # We need enough entropy to generate a seed.
    # Let's feed it multiple times if needed.
    for _ in range(50):
        n_produced = extractor.work(input_items, output_items)
        if n_produced > 0:
            break

    assert len(extractor.seed_history) > 0
    assert extractor.seed_history[0]['phi_rf'] > 0
    assert len(extractor.seed_history[0]['seed']) == 64 # Hex string of 32 bytes

def test_von_neumann_debias():
    extractor = ArkheEntropyExtractor()
    bits = [0, 0, 0, 1, 1, 0, 1, 1]
    # pairs: (0,0) discard, (0,1) -> 0, (1,0) -> 1, (1,1) discard
    expected = [0, 1]
    result = extractor._von_neumann_debias(bits)
    assert result == expected
