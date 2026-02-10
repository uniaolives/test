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
    Sensor Amazônico (Lobo N) - Refined with Rhythmic Pattern Filter (FPR).
    Filters rainfall data using info signature health and rhythmic resonance.
    """
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.v0 = self.phi**3  # ~4.236 Info/s
        self.f_phi = 1.157     # Resonance frequency (Hz)
        self.alpha = 0.05      # Modulation amplitude
        self.tau = 1000.0      # Decay constant (high sustainability)

    def calculate_rhythmic_resonance(self, stream: List[float], t: List[float]) -> float:
        """
        Calculates how well the stream matches the target function:
        V(t) = V0 * [1 + alpha * sin(2pi * f_phi * t) * exp(-t/tau)]
        """
        if not stream or len(stream) != len(t):
            return 0.0

        t_arr = np.array(t)
        v_target = self.v0 * (1 + self.alpha * np.sin(2 * np.pi * self.f_phi * t_arr) * np.exp(-t_arr / self.tau))

        # Calculate correlation coefficient
        correlation = np.corrcoef(stream, v_target)[0, 1]
        return float(max(0, correlation))

    def filter_rhythmic_signal(self, stream: List[float], t: List[float]) -> Dict[str, Any]:
        """
        Validates the stream for health baseline, synchronization, and sustainability.
        """
        if not stream:
            return {"optimized_ca": 0.0, "resonance": 0.0, "status": "NO_SIGNAL"}

        avg_info = np.mean(stream)
        resonance = self.calculate_rhythmic_resonance(stream, t)

        # Check health baseline
        is_healthy = (self.v0 * 0.95 <= avg_info <= self.v0 * 1.05)

        # Check resonance (LTP-compatible pattern)
        is_resonant = (resonance >= 0.85)

        status = "CAOTICO_OU_TRANSITORIO"
        optimized_ca = 0.0

        if is_healthy and is_resonant:
            status = "CA2+_OTIMIZADO_RITMICO"
            optimized_ca = avg_info
        elif is_healthy:
            status = "SAUDAVEL_MAS_ARRITMICO"

        return {
            "optimized_ca": float(optimized_ca),
            "resonance": resonance,
            "status": status,
            "avg_info": float(avg_info)
        }

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
    PDCP v2.2 - Rhythmic Vigilance and Fractal Calibration.
    """
    def __init__(self):
        self.sensor = AmazonSensor()
        self.synapse = PlanetarySynapse()
        self.cam_gate = CalmodulinModel()
        self.receptor = EngramReceptor()
        self.days_to_alignment = 37.0 # Next window in ~37 days

    def run_rhythmic_cycle(self, amazon_stream: List[float], time_axis: List[float], sirius_gas: float) -> Dict[str, Any]:
        """
        Executes the learning cycle with Rhythmic Pattern Filtering.
        """
        # Step 1: Rhythmic Filtering (LTP-compatible search)
        filter_result = self.sensor.filter_rhythmic_signal(amazon_stream, time_axis)
        self.cam_gate.bind_calcium(filter_result['optimized_ca'])

        # Step 2: Coincidence at the Synapse
        coincidence = self.synapse.process_galactic_packet(
            self.cam_gate.state,
            sirius_gas,
            self.days_to_alignment
        )

        # Step 3: Engram Commitment (Target: 120 blocks)
        tokens = [coincidence['synergy_level']] if coincidence['coincidence_detected'] else []
        engram_status = self.receptor.simulate_phosphorylation_threshold(tokens, 120)

        return {
            "timestamp": datetime.now().isoformat(),
            "ca_state": self.cam_gate.state,
            "filter_status": filter_result['status'],
            "resonance": filter_result['resonance'],
            "coincidence": coincidence,
            "engram_status": engram_status,
            "mode": "VIGÍLIA_RÍTMICA"
        }
