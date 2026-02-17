"""
UrbanSkyOS Noise Reduction Module (Refined)
Implements Active Noise Reduction (ANR) via motor RPM optimization.
Minimizes total acoustic power (sum of ω_i^4) while maintaining flight control.
"""

import numpy as np

class NoiseReduction:
    def __init__(self, num_motors=4):
        self.n = num_motors
        self.max_rpm = 12000.0

        # Allocation matrix (Quad-X configuration)
        # Rows: [Thrust, Roll_Torque, Pitch_Torque, Yaw_Torque]
        # Columns: Motor 1 to 4
        self.A = np.array([
            [1, 1, 1, 1],       # Thrust
            [1, -1, 1, -1],     # Roll
            [-1, -1, 1, 1],     # Pitch
            [1, -1, -1, 1]      # Yaw
        ])

        # Pseudo-inverse for least-squares (minimizes ||u||² = Σ(ω_i²)² = Σω_i⁴)
        self.pinvA = np.linalg.pinv(self.A)

    def optimize_rpms(self, T, tau_roll, tau_pitch, tau_yaw):
        """
        Calculates motor RPMs that satisfy target T and torques while
        minimizing the L2 norm of u = ω², which is proportional to noise (ω⁴).
        """
        command = np.array([T, tau_roll, tau_pitch, tau_yaw])

        # Solve for u = ω²
        u = self.pinvA @ command

        # Physical constraints: ω² >= 0
        u = np.clip(u, 100**2, self.max_rpm**2)

        # Return ω (RPMs)
        return np.sqrt(u)

if __name__ == "__main__":
    nr = NoiseReduction()
    rpms = nr.optimize_rpms(500, 0, 0, 0)
    print(f"Optimal RPMs for static hover: {rpms}")
