import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pyfrontend import GraphCaptureLayer, EnergyAnalyzer, AIConstitution, CCIRLoweringLayer

def run_constitutional_pipeline(source_code: str, model_name: str = None):
    print("\n🚀 COGNITIVE BRIDGE - PIPELINE CONSTITUCIONAL")
    capture_layer = GraphCaptureLayer()
    model_info = capture_layer.capture_from_source(source_code, model_name)

    analyzer = EnergyAnalyzer()
    energy_profile = analyzer.analyze_graph(model_info['graph_module'], model_info['input_shape'])

    constitution = AIConstitution()
    verification = constitution.verify_model(model_info, energy_profile)

    if not verification.passed:
        print("❌ REJEIÇÃO CONSTITUCIONAL")
        for v in verification.violations:
            print(f"   - {v.rule}: {v.message}")
        return {'status': 'REJECTED', 'violations': verification.violations, 'energy_profile': energy_profile}

    print("✅ APROVADO CONSTITUCIONALMENTE")
    lowerer = CCIRLoweringLayer()
    ccir = lowerer.lower(model_info['graph_module'], energy_profile, model_info['constitutional_metadata'], model_info['class_name'])
    print(f"   ✅ CCIR gerado ({len(ccir.splitlines())} linhas)")

    return {'status': 'APPROVED', 'energy_profile': energy_profile, 'ccir': ccir}

if __name__ == "__main__":
    dangerous_code = """
import torch.nn as nn
# constitutional:energy_budget=0.5
class EnergyViolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10000, 10000)
    def forward(self, x):
        return self.fc(x)
"""
    print("DEMO 1: REJEIÇÃO (Budget 0.5J, Model ~4J)")
    run_constitutional_pipeline(dangerous_code, 'EnergyViolator')

    efficient_code = """
import torch.nn as nn
# constitutional:energy_budget=1.0
class EfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)
"""
    print("\nDEMO 2: APROVAÇÃO (Budget 1.0J, Model < 0.1J)")
    run_constitutional_pipeline(efficient_code, 'EfficientModel')
