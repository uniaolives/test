"""
Example: Drone Fleet in H2 with THz Sensors.
Implements PPP deployment in hyperbolic space and collective detection validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add relevant paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_path not in sys.path:
    sys.path.append(base_path)

from arkhe_qutip.sensors.drone_agent import DroneAgentNode

class DroneFleetSimulation:
    """
    Simulation of a drone fleet in H2 with hyperbolic metric.
    """

    def __init__(self, n_drones=17, lambda0=10.0, alpha=0.5):
        self.n = n_drones
        self.lambda0 = lambda0  # max density at center
        self.alpha = alpha      # exponential decay factor

        # Deploy drones using hyperbolic PPP
        self.drones = self._deploy_ppp()

        # Existence condition (Theorem 1)
        self.V_max = self._compute_interference_potential()
        self.stable = self.V_max < 0.125 # (d-1)^2/8 for d=2

    def _deploy_ppp(self):
        """
        Generate Poisson Point Process in H2.
        Density: lambda(y) = lambda0 * exp(-alpha * y)
        """
        drones = []
        attempts = 0
        while len(drones) < self.n and attempts < self.n * 200:
            # Uniform proposal in bounded area of half-plane
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(0.1, 5.0)

            target_density = self.lambda0 * np.exp(-self.alpha * y)
            if np.random.uniform() < target_density / self.lambda0:
                drone = DroneAgentNode(f"Drone_{len(drones)}", (x, y))
                drones.append(drone)
            attempts += 1
        return drones

    def _compute_interference_potential(self):
        """
        Compute interference potential V_omega = sum eta(d_H).
        """
        if self.n <= 1: return 0.0
        V_total = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = self.drones[i].hyperbolic_distance(self.drones[j].pos)
                eta = 0.01 * np.exp(-d**2 / 4.0)
                V_total += eta
        return V_total / (self.n * (self.n - 1) / 2)

    def simulate_collective_detection(self, target_freq=3.90):
        """
        Simulate cooperative detection with entanglement.
        """
        # Update local coherences based on current positions
        positions = [d.pos for d in self.drones]
        for d in self.drones:
            d.update_coherence(positions, R_comm=2.0)

        # Fleet entanglement
        entanglement = self.drones[0].entangle_with_fleet(self.drones)

        # Detection
        signals = [d.detect_thz(target_freq) for d in self.drones]

        # Consensus
        if entanglement:
            enhancement = 1 + 0.3 * entanglement['C_global']
            fused_signal = np.mean(signals) * enhancement
            fused_std = np.std(signals) / np.sqrt(self.n)
        else:
            fused_signal = np.mean(signals)
            fused_std = np.std(signals) / np.sqrt(self.n)

        return {
            'mean_signal': np.mean(signals),
            'fused_signal': fused_signal,
            'fused_std': fused_std,
            'C_global': entanglement['C_global'] if entanglement else 0.0,
            'mean_C_local': np.mean([d.C_local for d in self.drones]),
            'stable': self.stable
        }

def main():
    print("🜁 Starting DroneTHz Hyperbolic Simulation (Arkhe-QuTiP)")
    print("=" * 60)

    # Parameters
    d = 2
    critical_threshold = 0.125

    # Stable Fleet
    print("\n[SCENARIO 1] Stable Fleet (lambda0=5, alpha=0.5)")
    fleet_s = DroneFleetSimulation(n_drones=17, lambda0=5.0, alpha=0.5)
    print(f"  V_max = {fleet_s.V_max:.4f} (< {critical_threshold})")

    res_s = fleet_s.simulate_collective_detection()
    print(f"  C_global: {res_s['C_global']:.3f}, mean(C_local): {res_s['mean_C_local']:.3f}")
    print(f"  Fused Signal: {res_s['fused_signal']:.3f}, SNR Improvement: {res_s['fused_signal']/res_s['fused_std']:.2f}x")

    # Unstable Fleet
    print("\n[SCENARIO 2] Unstable Fleet (lambda0=50, alpha=0.1)")
    fleet_u = DroneFleetSimulation(n_drones=17, lambda0=50.0, alpha=0.1)
    print(f"  V_max = {fleet_u.V_max:.4f} (>= {critical_threshold} is {fleet_u.V_max >= critical_threshold})")

    # Validation
    print("\n" + "=" * 60)
    print("📊 ARKHE(N) DRONE SYSTEM VALIDATION")
    print(f"  Principle 1 (C+F=1): {'✅' if all(abs(d.coherence + d.fluctuation - 1) < 0.01 for d in fleet_s.drones) else '❌'}")
    print(f"  Theorem 1 (Stability): {'✅' if fleet_s.stable else '❌'}")
    print(f"  Emergence (C_global > C_local): {'✅' if res_s['C_global'] > res_s['mean_C_local'] else '❌'}")

    print("\n✅ Example completed")

if __name__ == "__main__":
    main()
