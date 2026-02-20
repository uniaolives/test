import numpy as np
import matplotlib.pyplot as plt

class ArkheRfSim:
    """
    Simulation of S-Band Carrier Recovery (PLL) and AGC
    Models Doppler shift compensation (~±50 kHz)
    """
    def __init__(self, fs=1e6, carrier_freq=2.2e9):
        self.fs = fs  # Sampling rate
        self.carrier_freq = carrier_freq

        # PLL Parameters (Proportional-Integral Loop Filter)
        self.phase_est = 0.0
        self.freq_est = 0.0
        self.alpha = 0.1  # Increased proportional gain
        self.beta = 0.005 # Increased integral gain

        # AGC Parameters
        self.gain = 1.0
        self.target_amplitude = 1.0
        self.agc_mu = 0.01

    def step(self, input_signal):
        """Processes one sample through AGC and PLL"""
        # 1. AGC
        scaled_signal = input_signal * self.gain
        error_amp = self.target_amplitude - np.abs(scaled_signal)
        self.gain += self.agc_mu * error_amp

        # 2. PLL (Phase Detector)
        # reference: exp(j * phase_est)
        reference = np.exp(1j * self.phase_est)

        # Phase error: angle between input and reference
        phase_error = np.angle(scaled_signal * np.conj(reference))

        # Loop Filter & VCO update
        # freq_est accumulates the frequency offset
        self.freq_est += self.beta * phase_error
        self.phase_est += self.freq_est + self.alpha * phase_error

        # Wrap phase
        self.phase_est = (self.phase_est + np.pi) % (2 * np.pi) - np.pi

        return scaled_signal * np.conj(reference), phase_error

def run_simulation(duration=0.1, doppler_shift=20e3): # Reduced shift for faster lock demo
    fs = 1e6
    t = np.arange(0, duration, 1/fs)

    # Input signal with Doppler and varying amplitude
    amplitude = 0.5 + 0.1 * np.sin(2 * np.pi * 10 * t)
    input_sig = amplitude * np.exp(1j * (2 * np.pi * doppler_shift * t + 0.5))

    sim = ArkheRfSim(fs=fs)

    outputs = []
    errors = []
    gains = []
    freq_ests = []

    for sample in input_sig:
        out, err = sim.step(sample)
        outputs.append(out)
        errors.append(err)
        gains.append(sim.gain)
        freq_ests.append(sim.freq_est)

    print(f"Final Gain: {sim.gain:.4f}")
    final_freq_hz = sim.freq_est * fs / (2 * np.pi)
    print(f"Final Frequency Estimate: {final_freq_hz:.2f} Hz")

    return t, input_sig, np.array(outputs), np.array(errors), np.array(gains), np.array(freq_ests)

if __name__ == "__main__":
    t, sig_in, sig_out, err, gains, freq_ests = run_simulation()

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t[:500], np.real(sig_in[:500]), label='In (Real)')
    plt.plot(t[:500], np.real(sig_out[:500]), label='Out (Locked)')
    plt.title("Signal Locking (AGC + PLL)")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, err)
    plt.title("Phase Error (radians)")

    plt.subplot(4, 1, 3)
    plt.plot(t, gains)
    plt.title("AGC Gain")

    plt.subplot(4, 1, 4)
    plt.plot(t, freq_ests * 1e6 / (2 * np.pi))
    plt.title("Frequency Estimate (Hz)")

    plt.tight_layout()
    plt.savefig("rf_sim_results.png")
    print("Simulation plot saved as rf_sim_results.png")
