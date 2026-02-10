"""
Planetary Homeostasis and PDCP (Planetary Data Calibration Protocol).
Calibrates the macro system using micro-biological mechanics (CaM/CaMKII/AC1).
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from ..biological.calmodulin import CalmodulinModel, AdenylateCyclase1, CaMKIIInteraction

class AmazonSensor:
    """
    Sensor Amazônico (Lobo N).
    Filters noisy rainfall data using info signature affinity (phi^3 ± 0.034 * phi).
    """
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.base_affinity = self.phi**3 # ~4.236 Info/s
        self.tolerance = 0.034 * self.phi # ~0.055

    def filter_signal(self, raw_data_stream: List[float]) -> float:
        """
        Refined for robustness: busca padrões naturais (phi^3 ± tolerance).
        """
        if not raw_data_stream: return 0.0
        avg_info = np.mean(raw_data_stream)

        # Robust filtering range
        lower_bound = self.base_affinity - self.tolerance
        upper_bound = self.base_affinity + self.tolerance

        if lower_bound <= avg_info <= upper_bound:
            return avg_info
        return 0.0

class PlanetarySynapse:
    """
    Models the Earth-Sirius interface where coincidence detection happens.
    """
    def __init__(self):
        self.ac1_network = AdenylateCyclase1()
        self.sync_window = 42.3 # Days to Sirius alignment

    def process_galactic_packet(self, cam_state: str, gas_packet: float, days_to_alignment: float) -> Dict[str, Any]:
        """
        Sirius signal (Gas) is amplified during the orbital alignment window.
        """
        # Orbital multiplier: max at window center
        orbital_efficiency = np.exp(-((days_to_alignment - 0)**2) / 20.0)
        amplified_gas = gas_packet * orbital_efficiency

        return self.ac1_network.process_coincidence(cam_state, amplified_gas)

class EngramReceptor:
    """
    Simulates the 'Irreversible Commit' of climate memory into the planetary synapse.
    Based on CaMKII Thr286 autophosphorylation and the WAVEFORM_Ω-PROTO.1.
    """
    def __init__(self):
        self.camkii_hard_drive = CaMKIIInteraction()
        self.memory_canal = "Amazonia-Guiana Current"
        self.activation_signature_profile = "WAVEFORM_Ω-PROTO.1"

    def simulate_phosphorylation_threshold(self, token_flow: List[float], duration_blocks: int) -> Dict[str, Any]:
        """
        Determines the threshold to lock memory as LTP (Long-Term Potentiation).
        Target: 120 ± 10 blocks.
        """
        # Robustness check: signals that are 'too efficient' are penalized
        # We want complex/natural dialogue, not a perfect mirror.
        efficiency_factor = 1.0
        if duration_blocks < 110:
            efficiency_factor = 0.7 # Too fast = premature saturation

        frequency = len(token_flow) / 10.0
        result = self.camkii_hard_drive.simulate_frequency_decoding(token_flow, frequency * efficiency_factor * 10.0)

        profile = None
        if result['memory_state'] == "PERMANENT_LTP":
            profile = {
                "signature": self.activation_signature_profile,
                "energy_info": "phi^3 ± 0.034phi",
                "duration": f"{duration_blocks} blocks",
                "lock_status": "Indelible"
            }

        return {
            "memory_canal": self.memory_canal,
            "commit_status": result['memory_state'],
            "phosphorylation_level": result['thr286_phosphorylation'],
            "profile": profile
        }

class PlanetaryDataCalibrationProtocol:
    """
    PDCP v2.1 - Fractal Calibration for Robust Coincidence.
    """
    def __init__(self):
        self.sensor = AmazonSensor()
        self.synapse = PlanetarySynapse()
        self.cam_gate = CalmodulinModel()
        self.receptor = EngramReceptor()
        self.days_to_alignment = 42.3

    def run_vigilance_cycle(self, amazon_stream: List[float], sirius_gas: float, simulated_duration: int = 120) -> Dict[str, Any]:
        """
        Executes the refined learning cycle.
        """
        # Step 1: Filter (phi^3 ± 0.034phi)
        optimized_ca = self.sensor.filter_signal(amazon_stream)
        self.cam_gate.bind_calcium(optimized_ca)

        # Step 2: Coincidence at the Synapse (synchronized with Sirius)
        coincidence = self.synapse.process_galactic_packet(
            self.cam_gate.state,
            sirius_gas,
            self.days_to_alignment
        )

        # Step 3: Robust Engram Commitment
        tokens = [coincidence['synergy_level']] if coincidence['coincidence_detected'] else []
        engram_status = self.receptor.simulate_phosphorylation_threshold(tokens, simulated_duration)

        return {
            "timestamp": datetime.now().isoformat(),
            "ca_state": self.cam_gate.state,
            "coincidence": coincidence,
            "engram_status": engram_status,
            "mode": "CALIBRAÇÃO_FRACTAL_CONCLUÍDA"
        }
