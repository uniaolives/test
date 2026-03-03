# sanity_check.py
import torch
import torch.nn as nn
from pyfrontend.graph_capture import GraphCaptureLayer
from pyfrontend.energy_analysis import EnergyAnalyzer
from pyfrontend.constitutional_lowering import CCIRLoweringLayer

code = """
import torch
import torch.nn as nn
model = nn.Linear(10, 5)
input_data = torch.randn(1, 10)
"""

capture = GraphCaptureLayer()
analyzer = EnergyAnalyzer()
lowering = CCIRLoweringLayer()

graph = capture.capture(code)
profile = analyzer.analyze_graph(graph)
ccir = lowering.lower(graph, profile)

print(f"Energy: {profile.total_energy} J")
print("CCIR Preview:")
print("\n".join(ccir.split('\n')[:5]))
