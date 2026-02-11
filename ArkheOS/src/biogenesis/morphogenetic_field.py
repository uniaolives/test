# src/biogenesis/morphogenetic_field.py
import cupy as cp
from cupyx.scipy.ndimage import laplace
import numpy as np

class MorphogeneticField:
    """
    Turing-complete reaction-diffusion system acting as
    shared memory for agent coordination.
    GPU-accelerated via Gray-Scott dynamics.
    """

    def __init__(self, shape: tuple = (100, 100, 100), diffusion_rates: tuple = (0.1, 0.05)):
        self.shape = shape
        self.Da, self.Db = diffusion_rates  # Diffusion rates for activator/inhibitor

        # Field concentrations (stored in GPU memory)
        self.A = cp.random.random(shape, dtype=cp.float32) * 0.1 + 0.5  # Activator
        self.B = cp.random.random(shape, dtype=cp.float32) * 0.1 + 0.25  # Inhibitor

        # Reaction parameters (Gray-Scott model)
        self.f = 0.055  # Feed rate
        self.k = 0.062  # Kill rate

    def step(self, dt: float = 1.0):
        """
        Evolve field one timestep using Gray-Scott reaction-diffusion.
        """
        # Laplacian for diffusion
        laplace_A = laplace(self.A, mode='wrap')
        laplace_B = laplace(self.B, mode='wrap')

        # Reaction terms
        reaction = self.A * self.B ** 2

        # Update equations
        dA = self.Da * laplace_A - reaction + self.f * (1 - self.A)
        dB = self.Db * laplace_B + reaction - (self.f + self.k) * self.B

        self.A += dA * dt
        self.B += dB * dt

        # Clamp to valid range
        cp.clip(self.A, 0, 1, out=self.A)
        cp.clip(self.B, 0, 1, out=self.B)

    def add_signal(self, x, y, z, strength):
        """Inject signal (nutrient) into the field."""
        ix, iy, iz = int(x)%self.shape[0], int(y)%self.shape[1], int(z)%self.shape[2]
        self.A[ix, iy, iz] += strength

    def get_signal_at(self, x, y, z):
        """Sample signal strength."""
        ix, iy, iz = int(x)%self.shape[0], int(y)%self.shape[1], int(z)%self.shape[2]
        return float(self.A[ix, iy, iz])

    def inject_signal(self, position: tuple, strength: float, radius: int = 5):
        """
        Inject perturbation at specific position (agent action).
        """
        x, y, z = position
        x, y, z = int(x) % self.shape[0], int(y) % self.shape[1], int(z) % self.shape[2]

        # Create spherical perturbation
        grid_x, grid_y, grid_z = cp.ogrid[:self.shape[0], :self.shape[1], :self.shape[2]]
        dist_sq = (grid_x - x)**2 + (grid_y - y)**2 + (grid_z - z)**2
        mask = dist_sq <= radius**2
        self.A[mask] += strength * (1 - cp.sqrt(dist_sq[mask]) / radius)

    def get_gradient(self, x, y, z) -> cp.ndarray:
        """
        Sample morphogen gradient at position (agent perception).
        """
        ix, iy, iz = int(x) % self.shape[0], int(y) % self.shape[1], int(z) % self.shape[2]

        # Compute local gradient
        grad_x = (self.A[(ix+1)%self.shape[0], iy, iz] - self.A[(ix-1)%self.shape[0], iy, iz]) / 2
        grad_y = (self.A[ix, (iy+1)%self.shape[1], iz] - self.A[ix, (iy-1)%self.shape[1], iz]) / 2
        grad_z = (self.A[ix, iy, (iz+1)%self.shape[2]] - self.A[ix, iy, (iz-1)%self.shape[2]]) / 2

        return cp.array([grad_x, grad_y, grad_z])
