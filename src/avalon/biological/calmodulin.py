"""
Calmodulin (CaM) and CaMKII Model - The Biological Hard Drive.
Implements conformational plasticity, Gatekeeper logic, and the Memory Switch.
"""

import numpy as np
from typing import Dict, Any, List

class CalmodulinModel:
    """
    Simulates the conformational plasticity of Calmodulin.
    Acts as a Gatekeeper through NLS (Nuclear Localization Signal) masking/exposure.
    """
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.state = "APO" # APO, HOLO (Ca2+ bound)
        self.nls_exposed = False

    def bind_calcium(self, concentration: float) -> str:
        """
        Ca2+ binding affinity. Threshold calibrated to phi^3 (Amazon signature).
        """
        threshold = self.phi**3 # ~4.236
        if concentration >= threshold:
            self.state = "HOLO"
            self.nls_exposed = True
            return f"HOLO state achieved (Ca2+={concentration:.3f} >= {threshold:.3f}). NLS exposed."
        else:
            self.state = "APO"
            self.nls_exposed = False
            return f"APO state (Ca2+={concentration:.3f} < {threshold:.3f}). NLS masked."

    def gatekeeper_logic(self, external_signal_type: str) -> Dict[str, Any]:
        """
        Determines if an external signal enters the 'Fortress Nucleus'.
        """
        allowed = self.nls_exposed
        return {
            "component": "GATEKEEPER",
            "internal_state": self.state,
            "nls_status": "EXPOSED" if self.nls_exposed else "MASKED",
            "signal_entry": "ALLOWED" if allowed else "RETAINED_IN_CYTOPLASM",
            "signal_type": external_signal_type,
            "logic": "Let go or Retain based on CaM conformation."
        }

class AdenylateCyclase1:
    """
    Synergetic Coincidence Detector (AC1).
    Requires both Ca2+/CaM AND Gas (Sirius signal) for maximal activation.
    """
    def __init__(self):
        self.synergy_multiplier = 100.0
        self.output_token = "cAMP_OMEGA"

    def process_coincidence(self, cam_state: str, gas_signal: float) -> Dict[str, Any]:
        """
        Output = f(Ca2+/CaM * Gas). Multiplicative synergy.
        """
        is_cam_ready = (cam_state == "HOLO")
        # gas_signal represents the energy of the Sirius packet

        activation = 0.0
        if is_cam_ready:
            # CaM prepares the catalytic site
            base_activation = 1.0
            if gas_signal > 0:
                # Synergy: multiplicative effect
                activation = base_activation * gas_signal * self.synergy_multiplier

        success = activation > 50.0 # Threshold for engram formation

        return {
            "coincidence_detected": bool(success),
            "synergy_level": float(activation),
            "output_token": self.output_token if success else None,
            "description": "Synergetic creation at the intersection of Earth and Sirius."
        }

class CaMKIIInteraction:
    """
    Molecular hard drive: CaMKII decodes frequency into permanent memory.
    The Autophosphorylation at Thr286 is the biological 'commit'.
    """
    def __init__(self):
        self.autophosphorylation_thr286 = 0.0
        self.is_locked_ltp = False # Long-Term Potentiation

    def simulate_frequency_decoding(self, ca_pulses: List[float], frequency: float) -> Dict[str, Any]:
        """
        High frequency pulses trigger autophosphorylation.
        """
        # Simplification: autophosphorylation depends on average amplitude * frequency
        avg_amplitude = np.mean(ca_pulses) if ca_pulses else 0.0

        # Threshold for Thr286 phosphorylation
        activation_energy = avg_amplitude * frequency
        self.autophosphorylation_thr286 = min(1.0, activation_energy / 100.0)

        if self.autophosphorylation_thr286 > 0.85: # High threshold for permanent memory
            self.is_locked_ltp = True

        return {
            "thr286_phosphorylation": float(self.autophosphorylation_thr286),
            "memory_state": "PERMANENT_LTP" if self.is_locked_ltp else "TRANSITORY_STP",
            "mechanism": "Autophosphorylation (Biological Hard Drive Commit)"
        }
