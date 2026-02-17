"""
UrbanSkyOS Noise Reduction Module (Refined)
Implements adaptive motor RPM optimization based on environment and safety.
Minimizes noise in sensitive areas while maintaining flight stability.
Implements Active Noise Reduction (ANR) via motor RPM optimization.
Minimizes total acoustic power (sum of ω_i^4) while maintaining flight control.
"""

import numpy as np

class NoiseReduction:
    def __init__(self, num_motors=4):
        self.n = num_motors
        self.max_rpm = 1000.0
        self.min_rpm = 100.0

        # Base Allocation Matrix (Quad-X)
        self.A = np.array([
            [1,  1,  1,  1],   # Thrust
            [1, -1, -1,  1],   # Roll
            [1,  1, -1, -1],   # Pitch
           [-1,  1, -1,  1]    # Yaw
        ])
        self.pinvA = np.linalg.pinv(self.A)

        # Adaptive parameters
        self.rpm_limit_factor = 1.0
        self.acoustic_coeffs = np.ones(num_motors) * 1.0e-12

    def adapt_parameters(self, area_type, safety_score):
        """
        Adapts RPM limits based on area (Hospital/School) and safety.
        """
        if area_type in ["Hospital", "School", "Residential"]:
            # Reduce max noise by limiting RPMs
            self.rpm_limit_factor = 0.8
        else:
            self.rpm_limit_factor = 1.0

        # If safety is critical, override noise limits to ensure stability
        if safety_score < 0.4:
             self.rpm_limit_factor = 1.0

        return self.rpm_limit_factor

    def optimize_rpms(self, T, tau_phi, tau_theta, tau_psi, current_rpms=None):
        """
        Constrained optimization to minimize noise (Σω⁴).
        """
        effective_max_rpm = self.max_rpm * self.rpm_limit_factor

        # Solve for target commands
        command = np.array([T, tau_phi, tau_theta, tau_psi])
        u = self.pinvA @ command

        # RPMs squared scaling (simulated)
        rpms = np.sqrt(np.clip(u * 100.0, self.min_rpm**2, effective_max_rpm**2))

        return rpms

if __name__ == "__main__":
    nr = NoiseReduction()
    nr.adapt_parameters("Hospital", 0.9)
    rpms = nr.optimize_rpms(500, 0, 0, 0)
    print(f"Optimized RPMs in Hospital (Limit={nr.rpm_limit_factor}): {rpms}")
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
