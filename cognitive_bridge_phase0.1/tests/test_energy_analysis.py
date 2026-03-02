import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule
from pyfrontend.graph_capture import GraphCaptureLayer
from pyfrontend.energy_analysis import EnergyAnalyzer

def test_energy_estimation_linear():
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 100)
        def forward(self, x):
            return self.fc(x)

    model = LinearModel()
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    gm = GraphModule(model, graph)

    analyzer = EnergyAnalyzer(hardware_target='GPU_NVIDIA_A100')
    profile = analyzer.analyze_graph(gm, (1, 100))

    assert profile.total_energy > 0
    assert profile.compute_energy > 0
    assert profile.memory_energy > 0
    assert profile.total_energy < 1.0
