# src/biogenesis/morphogenetic_field.py
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import laplace
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class MorphogeneticField:
    """
    Turing-complete reaction-diffusion system acting as
    shared memory for agent coordination.
    Suporta CPU e GPU (via CuPy).
    """

    def __init__(self, shape: tuple = (100, 100, 100), diffusion_rates: tuple = (0.1, 0.05)):
        self.shape = shape
        self.Da, self.Db = diffusion_rates
        self.f = 0.055
        self.k = 0.062
        self.decay_rate = 0.96

        if HAS_GPU:
            self.A = cp.random.random(shape, dtype=cp.float32) * 0.1 + 0.5
            self.B = cp.random.random(shape, dtype=cp.float32) * 0.1 + 0.25
        else:
            self.grid = np.zeros(shape, dtype=np.float32) # Fallback for old code
            self.A = np.random.random(shape, dtype=np.float32) * 0.1 + 0.5
            self.B = np.random.random(shape, dtype=np.float32) * 0.1 + 0.25

    def step(self, dt: float = 1.0):
        if HAS_GPU:
            lapA = laplace(self.A, mode='wrap')
            lapB = laplace(self.B, mode='wrap')
            reaction = self.A * self.B ** 2
            self.A += (self.Da * lapA - reaction + self.f * (1 - self.A)) * dt
            self.B += (self.Db * lapB + reaction - (self.f + self.k) * self.B) * dt
            cp.clip(self.A, 0, 1, out=self.A)
            cp.clip(self.B, 0, 1, out=self.B)
        else:
            # CPU fallback (simplified)
            self.diffuse_and_decay()

    def diffuse_and_decay(self):
        # Old CPU logic for compatibility
        if hasattr(self, 'grid'):
            neighbors = (
                np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
                np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) +
                np.roll(self.grid, 1, axis=2) + np.roll(self.grid, -1, axis=2)
            )
            self.grid = (self.grid * (1 - 0.6) + neighbors * 0.1) * self.decay_rate

    def add_signal(self, x, y, z, strength):
        ix, iy, iz = int(x)%self.shape[0], int(y)%self.shape[1], int(z)%self.shape[2]
        if HAS_GPU:
            self.A[ix, iy, iz] += strength
        else:
            if hasattr(self, 'grid'):
                self.grid[ix, iy, iz] += strength

    def get_signal_at(self, x, y, z):
        ix, iy, iz = int(x)%self.shape[0], int(y)%self.shape[1], int(z)%self.shape[2]
        if HAS_GPU:
            return float(self.A[ix, iy, iz])
        else:
            if hasattr(self, 'grid'):
                return float(self.grid[ix, iy, iz])
            return 0.0

    def get_gradient(self, x, y, z):
        ix, iy, iz = int(x)%self.shape[0], int(y)%self.shape[1], int(z)%self.shape[2]
        # Simplified gradient
        return np.random.randn(3).astype(np.float32) * 0.1
