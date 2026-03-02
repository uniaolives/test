"""
Circuit QED Measurement Simulation for Qubit Waveform Retrieval.
Models the readout of a superconducting qubit via a resonator.
"""
import numpy as np
import qutip as qt

class CircuitQEDReadout:
    """
    Simulates the readout of a qubit in a Circuit QED architecture.
    Uses the dispersive regime for non-destructive measurement.
    """
    def __init__(self, qubit_freq=5.0e9, resonator_freq=6.0e9, coupling_g=100.0e6):
        self.wq = qubit_freq * 2 * np.pi
        self.wr = resonator_freq * 2 * np.pi
        self.g = coupling_g * 2 * np.pi

        # Dispersive shift chi = g^2 / delta
        self.delta = self.wr - self.wq
        self.chi = (self.g**2) / self.delta

    def simulate_measurement(self, qubit_state, noise_temp=0.01):
        """
        Simulates the IQ signal output of a measurement pulse.
        """
        # Frequency shift depends on qubit state
        # wr_eff = wr +/- chi
        if qubit_state == '0':
            w_eff = self.wr - self.chi
        else:
            w_eff = self.wr + self.chi

        # IQ signal representation
        times = np.linspace(0, 1e-6, 1000) # 1 microsecond

        # Signal + thermal noise
        signal_i = np.cos(w_eff * times) + np.random.normal(0, noise_temp, len(times))
        signal_q = np.sin(w_eff * times) + np.random.normal(0, noise_temp, len(times))

        return times, signal_i, signal_q

    def extract_fidelity(self, retrieved_signal, original_buffer):
        """
        Calculates measurement fidelity using SNR analysis.
        """
        # Simplified fidelity calculation
        snr = np.var(original_buffer) / np.var(retrieved_signal - original_buffer)
        fidelity = 1.0 - (1.0 / (1.0 + snr))
        return fidelity

if __name__ == "__main__":
    readout = CircuitQEDReadout()
    t, i, q = readout.simulate_measurement('1')
    print(f"Readout signal generated for state '1' with {len(t)} points.")
