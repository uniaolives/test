import pytest
import awkward as ak
import numpy as np
import os
from gateway.app.physics.arkhe_s2_lhc import ArkheLHCTrigger

def test_s2_trigger_logic():
    trigger = ArkheLHCTrigger(time_resolution=1e-13, cone_size=0.4)

    # Create a mock event with awkward
    # 2 jets, one with negative dt relative to the other
    event = ak.Array([{
        'jet_pt': [100.0, 100.0],
        'jet_eta': [0.1, 0.12],
        'jet_phi': [0.1, 0.12],
        'jet_time': [1e-12, 2e-12], # i=0 "earlier" than i=1 (dt = -1e-12)
    }])[0]

    res = trigger.compute_arkhe_score(event)
    assert res['n_violations'] == 1
    assert res['arkhe_score'] > 0

def test_s2_filter_events():
    trigger = ArkheLHCTrigger(time_resolution=1e-13)
    events = ak.Array([
        {
            'jet_pt': [100.0, 100.0],
            'jet_eta': [0.1, 0.12],
            'jet_phi': [0.1, 0.12],
            'jet_time': [1e-12, 2e-12], # Violator
        },
        {
            'jet_pt': [50.0, 50.0],
            'jet_eta': [0.5, 2.0],
            'jet_phi': [0.5, 2.0],
            'jet_time': [1e-12, 1.0001e-12], # Honest (small dt but large dR)
        }
    ])

    filtered = trigger.filter_events(events, threshold=0.001)
    assert len(filtered) == 1
    assert filtered[0].arkhe_n_violations == 1
