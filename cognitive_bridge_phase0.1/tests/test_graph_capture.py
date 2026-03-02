import pytest
import torch
import torch.nn as nn
from pyfrontend.graph_capture import GraphCaptureLayer

def test_capture_from_source():
    source_code = """
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
"""
    capture = GraphCaptureLayer()
    model_info = capture.capture_from_source(source_code, "SimpleModel")

    assert model_info['class_name'] == "SimpleModel"
    assert model_info['parameter_count'] == 10 * 5 + 5
    assert 'graph_module' in model_info

def test_extract_metadata():
    source_code = """
# constitutional:energy_budget=0.7
# constitutional:max_parameters=5000
class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)
"""
    capture = GraphCaptureLayer()
    model_info = capture.capture_from_source(source_code, "MetaModel")

    metadata = model_info['constitutional_metadata']
    assert metadata['energy_budget'] == 0.7
    assert metadata['max_parameters'] == 5000
