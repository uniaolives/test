"""
PYFRONTEND - Ponte Python → CCIR
Captura modelos PyTorch e aplica verificação constitucional
"""

from .graph_capture import GraphCaptureLayer
from .energy_analysis import EnergyAnalyzer, EnergyProfile
from .constitutional_lowering import CCIRLoweringLayer
from .constitution import AIConstitution, verify_constitutional_compliance

__version__ = "0.1.0"
__all__ = [
    'GraphCaptureLayer',
    'EnergyAnalyzer',
    'EnergyProfile',
    'CCIRLoweringLayer',
    'AIConstitution',
    'verify_constitutional_compliance'
]
