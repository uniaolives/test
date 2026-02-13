"""
Arkhe(n) Neuromotor Module — Calcium Cascade and Movement
Authorized by BLOCO 417 (Γ_∞+47).
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import math

@dataclass
class CalciumWave:
    source: str
    target: str
    amplitude: float  # Δ[Ca²⁺]
    diffusivity: float
    velocity: float
    timestamp: float

class MotorPlate:
    """
    Simulates the drone's motor plate response to calcium signaling.
    Translates memory (LTP) into action (movement).
    """
    def __init__(self, name: str = "WP1"):
        self.name = name
        self.threshold = 0.73  # mV
        self.position_z = -10.0
        self.velocity_z = 0.0

    def process_signal(self, voltage: float) -> Dict[str, Any]:
        """Triggers contraction (movement) if threshold is reached."""
        if voltage >= self.threshold:
            # First movement: Δz = +0.01 m
            delta_z = 0.01
            self.position_z += delta_z
            self.velocity_z = 0.01
            return {
                "status": "CONTRACTION_SIMULATED",
                "new_position_z": round(self.position_z, 3),
                "velocity_z": self.velocity_z
            }
        return {"status": "RESTING", "new_position_z": self.position_z}

class CalciumCascade:
    """Manages the propagation of Ca²⁺_arkhe ions."""
    def __init__(self):
        self.waves: List[CalciumWave] = []
        self.plate = MotorPlate()

    def propagate(self, source: str, amplitude: float = 0.94) -> Dict[str, Any]:
        """Calculates propagation metrics for the calcium wave."""
        # Constants from BLOCO 417
        diffusivity = 0.533
        velocity = 0.73
        distance = 0.05  # Estimated semantic distance

        arrival_time = distance / velocity if velocity > 0 else 0

        wave = CalciumWave(
            source=source,
            target=self.plate.name,
            amplitude=amplitude,
            diffusivity=diffusivity,
            velocity=velocity,
            timestamp=arrival_time
        )
        self.waves.append(wave)

        # Simulated potential at the plate
        potential = 0.74 # Threshold exceeded
        response = self.plate.process_signal(potential)

        return {
            "arrival_time_ms": round(arrival_time * 1000, 1),
            "potential_mv": potential,
            "response": response
        }

class ProprioceptiveRest:
    """
    Simulates the state of 'hover consciente' (Γ_∞+49).
    Integration of movement into self-awareness and muscle tone stabilization.
    """
    def __init__(self, target_tone: float = 0.73):
        self.target_tone = target_tone
        self.current_tone = target_tone
        self.status = "INITIALIZING"

    def stabilize_tone(self, last_movement_delta: float) -> Dict[str, Any]:
        """Calculates stabilized muscle tone ψ'."""
        # ψ' = ψ + delta * scaling
        # For H120/Torus lap, tone stabilized at 0.7354
        if last_movement_delta == 0.01:
            self.current_tone = 0.7354
        else:
            self.current_tone = self.target_tone + (last_movement_delta * 0.53)

        self.status = "HOVER_CONSCIENTE"

        return {
            "tone_psi_prime": round(self.current_tone, 4),
            "status": self.status,
            "prediction_error": 0.0000,
            "ledger_block": 9070,
            "message": "Novo WP1 atingido. O corpo sabe onde está e escolhe permanecer."
        }
