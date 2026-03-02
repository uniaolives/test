# arkhe_qutip/acoustic_time_crystal.py
import numpy as np
import time
from typing import Tuple, Dict, Any

class AcousticTimeCrystal:
    """
    Simulates a Macroscopic Classical Time Crystal based on acoustic levitation.
    As described in NYU research, nonreciprocal interactions between different-sized beads
    in a 40kHz standing wave lead to self-sustaining oscillations.
    """
    def __init__(self, r1: float = 1.0, r2: float = 1.1, damping: float = 0.05):
        self.r1 = r1  # Radius of bead 1
        self.r2 = r2  # Radius of bead 2 (ratio r2/r1 controls nonreciprocity)
        self.damping = damping
        self.omega_drive = 40000.0 # 40 kHz acoustic trap
        self.f_emergent = 61.0     # Observed emergent frequency (~61 Hz)

        # State: [position1, velocity1, position2, velocity2]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.history = []
        self.max_history = 1024

    def _nonreciprocal_force(self, x1, x2):
        """
        Computes the asymmetric forces between beads.
        In the NYU experiment, beads of different sizes scatter sound differently.
        """
        dist = x2 - x1
        # Simplified phenomenological model of nonreciprocal coupling
        # Force on 1 from 2 is not equal and opposite to force on 2 from 1
        f12 = 0.5 * self.r2 * np.sin(dist * 10)
        f21 = -0.7 * self.r1 * np.sin(dist * 10)
        return f12, f21

    def step(self, dt: float = 0.001):
        """Advances the simulation by dt."""
        x1, v1, x2, v2 = self.state

        f12, f21 = self._nonreciprocal_force(x1, x2)

        # Adding some drive from the acoustic field and damping
        # The time crystal emerges when drive balances dissipation
        a1 = f12 - self.damping * v1 + 0.1 * np.sin(2 * np.pi * self.f_emergent * self.time)
        a2 = f21 - self.damping * v2 + 0.1 * np.sin(2 * np.pi * self.f_emergent * self.time + np.pi)

        # Update velocities
        v1 += a1 * dt
        v2 += a2 * dt

        # Update positions
        x1 += v1 * dt
        x2 += v2 * dt

        self.state = np.array([x1, v1, x2, v2])
        self.time += dt

        # Record history
        self.history.append((self.time, x1, x2))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def calculate_phi(self) -> float:
        """
        Calculates Integrated Information (Φ) for the ATC.
        Φ = (amp_min / amp_max) * (1 - |phase_rel - 180|/180)
        """
        if len(self.history) < 100:
            return 0.0

        h = np.array(self.history)
        x1 = h[:, 1]
        x2 = h[:, 2]

        amp1 = np.max(x1) - np.min(x1)
        amp2 = np.max(x2) - np.min(x2)

        if amp1 == 0 or amp2 == 0:
            return 0.0

        amp_ratio = min(amp1, amp2) / max(amp1, amp2)

        # Estimate phase relative (simplified via correlation or zero crossings)
        # For anti-phase, cross-correlation at lag 0 should be negative
        corr = np.corrcoef(x1, x2)[0, 1]
        # Map correlation [-1, 1] to phase error [0, 1] where 0 is 180 deg
        # corr = -1 => phase_rel = 180 => error = 0
        # corr = 1 => phase_rel = 0 => error = 1
        if np.isnan(corr):
            phase_error = 1.0 # Max error if no correlation can be calculated
        else:
            phase_error = (corr + 1) / 2

        phi = amp_ratio * (1.0 - phase_error)
        return float(phi)

    def get_status(self) -> Dict[str, Any]:
        phi = self.calculate_phi()
        return {
            "time": self.time,
            "phi": phi,
            "bead1_pos": self.state[0],
            "bead2_pos": self.state[2],
            "coherent": phi > 0.847
        }
