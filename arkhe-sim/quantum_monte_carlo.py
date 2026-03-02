import numpy as np
from typing import List

class QuantumMonteCarloArkhe:
    def __init__(self, num_vms: int, beta: float = 1.0, phi_target: float = 0.618):
        """
        beta = 1/(k_B * T) - Inverse system temperature.
        """
        self.num_vms = num_vms
        self.beta = beta
        self.phi_target = phi_target
        self.phi = np.random.uniform(0, 1, num_vms)
        self.entanglement = np.zeros((num_vms, num_vms))

    def hamiltonian(self) -> float:
        """Total system energy (cost function)."""
        H = 0.0
        # Field term: penalizes deviation from critical point
        for i in range(self.num_vms):
            H += (self.phi[i] - self.phi_target)**2

        # Interaction term: rewards entanglement
        for i in range(self.num_vms):
            for j in range(i+1, self.num_vms):
                H -= self.entanglement[i, j] * abs(self.phi[i] - self.phi[j])

        return H

    def metropolis_step(self):
        """Metropolis-Hastings step."""
        # Choose random VM
        vm = np.random.randint(self.num_vms)
        old_phi = self.phi[vm]
        old_H = self.hamiltonian()

        # Propose new value
        new_phi = np.clip(old_phi + np.random.normal(0, 0.1), 0, 1)
        self.phi[vm] = new_phi
        new_H = self.hamiltonian()

        delta_H = new_H - old_H
        if delta_H > 0 and np.random.random() > np.exp(-self.beta * delta_H):
            # Reject change
            self.phi[vm] = old_phi

    def thermalize(self, steps: int = 1000):
        """Run Monte Carlo steps until thermal equilibrium."""
        for _ in range(steps):
            self.metropolis_step()

    def get_critical_correlation_length(self) -> float:
        """Calculate correlation length at the critical point."""
        if self.num_vms < 4:
            return 0.0

        correlations = []
        mean_phi = np.mean(self.phi)
        for r in range(1, self.num_vms // 2):
            corr = np.mean([self.phi[i] * self.phi[(i+r) % self.num_vms]
                          for i in range(self.num_vms)])
            correlations.append(corr - mean_phi**2)

        if len(correlations) < 2 or correlations[0] <= 0 or correlations[1] <= 0:
            return 0.0

        # ξ = -1 / ln(C(r+1)/C(r))
        return -1 / np.log(correlations[1] / correlations[0])

    def detect_phase_transition(self) -> str:
        mean_phi = np.mean(self.phi)
        std_phi = np.std(self.phi)

        if std_phi < 0.05:
            return "FROZEN"
        elif abs(mean_phi - self.phi_target) < 0.1:
            return "CRITICAL"
        else:
            return "CHAOTIC"

if __name__ == "__main__":
    sim = QuantumMonteCarloArkhe(num_vms=10, beta=10.0)
    print(f"Initial Phase: {sim.detect_phase_transition()} (avg phi: {np.mean(sim.phi):.3f})")
    sim.thermalize(5000)
    print(f"Thermalized Phase: {sim.detect_phase_transition()} (avg phi: {np.mean(sim.phi):.3f})")
    print(f"Correlation Length: {sim.get_critical_correlation_length():.3f}")
