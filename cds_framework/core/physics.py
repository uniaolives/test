# cds_framework/core/physics.py
import numpy as np

class PhiFieldSimulator:
    """
    Core Physics Engine for the Consciousness Dynamics Simulator (CDS).
    Implements Time-Dependent Landau-Ginzburg (TDLG) dynamics for the order parameter Φ.
    """
    def __init__(self, size=100, r=-1.0, u=1.0, gamma=1.0, dt=0.01, dx=1.0):
        self.size = size
        self.r = r  # Linear coefficient (negative for symmetry breaking)
        self.u = u  # Quartic coefficient (positive for stability)
        self.gamma = gamma  # Kinetic coefficient
        self.dt = dt
        self.dx = dx

        # Initialize Φ field with small noise
        self.phi = np.random.normal(0, 0.01, size)

    def free_energy_density(self):
        """Calculates the local free energy density."""
        return 0.5 * self.r * self.phi**2 + 0.25 * self.u * self.phi**4

    def compute_laplacian(self):
        """Computes the Laplacian of the Φ field using central differences."""
        phi_left = np.roll(self.phi, 1)
        phi_right = np.roll(self.phi, -1)
        return (phi_left + phi_right - 2 * self.phi) / (self.dx**2)

    def step(self, external_h=0.0):
        """
        Performs one time step of the TDLG evolution:
        dΦ/dt = -Γ * (δF/δΦ) = -Γ * (-∇²Φ + rΦ + uΦ³ - H)
        """
        laplacian = self.compute_laplacian()
        variation = -laplacian + self.r * self.phi + self.u * self.phi**3 - external_h
        d_phi = -self.gamma * variation * self.dt
        self.phi += d_phi
        return self.phi

    def get_order_parameter(self):
        """Returns the global order parameter (average Φ)."""
        return np.mean(self.phi)

    def simulate(self, steps=1000, external_h=0.0):
        """Runs the simulation for a number of steps."""
        history = []
        for _ in range(steps):
            self.step(external_h)
            history.append(self.get_order_parameter())
        return np.array(history)
