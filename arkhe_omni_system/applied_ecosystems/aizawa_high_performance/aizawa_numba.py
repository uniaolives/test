import numpy as np
from numba import njit, prange
import time

@njit(inline='always')
def aizawa_rk4_step(x, y, z, dt):
    # Parameters
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1

    def derivatives(x, y, z):
        dx = (z - b) * x - d * y
        dy = d * x + (z - b) * y
        dz = c + a * z - (z**3)/3.0 - (x**2 + y**2) * (1.0 + e * z) + f * z * (x**3)
        return dx, dy, dz

    k1x, k1y, k1z = derivatives(x, y, z)

    k2x, k2y, k2z = derivatives(x + 0.5 * dt * k1x,
                                y + 0.5 * dt * k1y,
                                z + 0.5 * dt * k1z)

    k3x, k3y, k3z = derivatives(x + 0.5 * dt * k2x,
                                y + 0.5 * dt * k2y,
                                z + 0.5 * dt * k2z)

    k4x, k4y, k4z = derivatives(x + dt * k3x,
                                y + dt * k3y,
                                z + dt * k3z)

    new_x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    new_y = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    new_z = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

    return new_x, new_y, new_z

@njit(parallel=True)
def simulate_aizawa_ensemble(xs, ys, zs, steps, dt):
    n = xs.shape[0]
    for i in prange(n):
        x, y, z = xs[i], ys[i], zs[i]
        for _ in range(steps):
            x, y, z = aizawa_rk4_step(x, y, z, dt)
        xs[i], ys[i], zs[i] = x, y, z

def main():
    n_points = 1_000_000
    steps = 100
    dt = 0.01

    print(f"Starting Aizawa Ensemble Simulation with {n_points} points for {steps} steps...")

    xs = np.random.randn(n_points).astype(np.float64)
    ys = np.random.randn(n_points).astype(np.float64)
    zs = np.random.randn(n_points).astype(np.float64)

    # Warm up
    simulate_aizawa_ensemble(xs[:100], ys[:100], zs[:100], 1, dt)

    start_time = time.time()
    simulate_aizawa_ensemble(xs, ys, zs, steps, dt)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Simulation completed in {duration:.4f} seconds.")
    print(f"Performance: {n_points * steps / duration / 1e6:.2f} million iterations per second.")

if __name__ == "__main__":
    main()
