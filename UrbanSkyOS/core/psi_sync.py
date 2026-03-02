"""
UrbanSkyOS Psi Synchronization
Central synchronization for all subsystems at 40Hz (gamma band).
Implements the Ψ-layer as a real-time scheduler simulation.
"""

import time
import numpy as np

class PsiSync:
    def __init__(self, base_freq=40.0):
        self.base_frequency = base_freq
        self.period = 1.0 / self.base_frequency

        # Subsystem phases
        self.phases = {
            'lidar': 0.0,
            'venus_tx': 0.25,
            'control': 0.5,
            'noise_opt': 0.75,
            'federation': 0.9
        }

        self.cycle_coherence = 1.0
        self.phase_jitter = 0.0
        self.start_time = time.time()

    def get_sleep_time(self, subsystem, cycle_start):
        """Calculates time to sleep until next phase."""
        target_time = cycle_start + self.phases[subsystem] * self.period
        return max(0, target_time - time.time())

    def update_coherence(self, actual_period):
        """Updates cycle coherence based on jitter."""
        self.phase_jitter = abs(actual_period - self.period)
        self.cycle_coherence = np.exp(-self.phase_jitter / self.period)

        if self.cycle_coherence < 0.847:
             self.period = min(0.05, self.period * (1.5 - self.cycle_coherence))
             self.base_frequency = 1.0 / self.period

        return self.cycle_coherence
