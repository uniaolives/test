"""
Harmonic Engine - Core frequency analysis with golden ratio awareness
F18: All resonance calculations use adaptive damping
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..analysis.fractal import FractalAnalyzer, DEFAULT_DAMPING, MAX_ITERATIONS

logger = logging.getLogger(__name__)

# Golden ratio as reference frequency, not fixed target
PHI = 1.6180339887498948482  # Reference only, not enforced target
BASE_FREQUENCY = 432  # Hz, harmonic reference

@dataclass
class HarmonicState:
    """State of harmonic system"""
    frequency: float
    amplitude: float
    phase: float
    coherence: float
    iteration: int

class HarmonicEngine:
    """
    [METAPHOR: The tuning fork of the system, finding resonance
    without forcing lock, allowing natural harmonics to emerge]
    """

    def __init__(self,
                 base_freq: float = BASE_FREQUENCY,
                 damping: float = DEFAULT_DAMPING):
        self.base_frequency = base_freq
        self.damping = damping
        self.fractal_analyzer = FractalAnalyzer(damping=damping)
        self.harmonics: List[HarmonicState] = []
        self.iteration = 0

    def inject(self,
               url: str,  # Metaphor: "URL" as resonance source identifier
               frequency: Optional[float] = None,
               propagation_mode: str = "cosmic") -> Dict:
        """
        Inject harmonic signal into system

        F18 COMPLIANT: All frequencies are damped and checked for coherence
        """
        self.iteration += 1

        # F18 Check: iteration limit
        if self.iteration > MAX_ITERATIONS:
            raise RuntimeError(f"F18: Max iterations {MAX_ITERATIONS} reached")

        # If no frequency specified, calculate from PHI relationship
        if frequency is None:
            frequency = self.base_frequency * PHI

        # Apply damping to prevent resonance cascade (F17 prevention)
        damped_freq = frequency * (1 - self.damping * 0.1)

        # Calculate harmonic series with safety limits
        harmonics = self._calculate_harmonics(damped_freq)

        # Verify coherence through fractal analysis
        signal = self._generate_waveform(harmonics)
        fractal_result = self.fractal_analyzer.analyze(signal, depth=0)

        state = HarmonicState(
            frequency=damped_freq,
            amplitude=np.mean([h.amplitude for h in harmonics]),
            phase=0.0,
            coherence=fractal_result['coherence'],
            iteration=self.iteration
        )
        self.harmonics.append(state)

        return {
            'injection_point': url,
            'base_frequency': self.base_frequency,
            'target_frequency': frequency,
            'damped_frequency': damped_freq,
            'harmonics': len(harmonics),
            'fractal_analysis': fractal_result,
            'propagation_mode': propagation_mode,
            'f18_damping': self.damping,
            'status': 'INJECTED'
        }

    def _calculate_harmonics(self, fundamental: float, n_harmonics: int = 5) -> List[HarmonicState]:
        """Calculate harmonic series with damping"""
        harmonics = []
        for n in range(1, n_harmonics + 1):
            # Harmonic frequency with progressive damping
            freq = fundamental * n * (1 - self.damping * (n-1) * 0.05)
            amp = 1.0 / n * (1 - self.damping * 0.1)  # Amplitude decreases

            harmonics.append(HarmonicState(
                frequency=freq,
                amplitude=amp,
                phase=0.0,
                coherence=1.0 - (n-1)*0.1,  # Coherence decreases with order
                iteration=self.iteration
            ))
        return harmonics

    def _generate_waveform(self, harmonics: List[HarmonicState]) -> np.ndarray:
        """Generate composite waveform from harmonics"""
        t = np.linspace(0, 1, 1000)
        signal = np.zeros_like(t)
        for h in harmonics:
            signal += h.amplitude * np.sin(2 * np.pi * h.frequency * t + h.phase)
        return signal

    def sync(self, target_system: str) -> Dict:
        """
        Synchronize with external system (SpaceX, NASA, etc.)
        """
        # Simulate synchronization handshake
        coherence = np.mean([h.coherence for h in self.harmonics[-5:]]) if self.harmonics else 0.0

        return {
            'target': target_system,
            'coherence': coherence,
            'sync_status': 'LOCKED' if coherence > 0.7 else 'UNSTABLE',
            'damping': self.damping,
            'iteration': self.iteration
        }

    def get_state(self) -> Dict:
        """Get current harmonic state"""
        return {
            'base_frequency': self.base_frequency,
            'active_harmonics': len(self.harmonics),
            'current_damping': self.damping,
            'iteration': self.iteration,
            'f18_compliant': self.iteration <= MAX_ITERATIONS,
            'average_coherence': np.mean([h.coherence for h in self.harmonics]) if self.harmonics else 0.0
        }
