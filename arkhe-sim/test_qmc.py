from quantum_monte_carlo import QuantumMonteCarloArkhe
import numpy as np

def test_qmc():
    print("Testing QuantumMonteCarloArkhe...")
    sim = QuantumMonteCarloArkhe(num_vms=10, beta=1.0)

    initial_h = sim.hamiltonian()
    print(f"Initial Hamiltonian: {initial_h:.4f}")

    sim.thermalize(steps=1000)

    final_h = sim.hamiltonian()
    print(f"Final Hamiltonian after thermalization: {final_h:.4f}")

    # In most cases, H should decrease as it tries to reach the target phi
    # although it might fluctuate due to beta=1.0

    phase = sim.detect_phase_transition()
    print(f"Detected Phase: {phase}")

    corr_len = sim.get_critical_correlation_length()
    print(f"Correlation Length: {corr_len:.4f}")

    assert 0 <= np.mean(sim.phi) <= 1
    print("Test passed!")

if __name__ == "__main__":
    test_qmc()
