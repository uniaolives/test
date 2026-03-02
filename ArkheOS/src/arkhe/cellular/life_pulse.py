"""
Life Pulse Generation
Simulating cellular heartbeat via PI harmonics
"""

import numpy as np

def generate_life_pulse(duration=60, fs=100):
    """
    Generate a life pulse signal representing cellular heartbeat.

    fundamental: 0.1 Hz (calcium wave)
    harmonics: based on PI concentrations
    """
    t = np.linspace(0, duration, int(fs*duration))

    # frequência fundamental: 0.1 Hz (onda de cálcio)
    fundamental = np.sin(2*np.pi*0.1*t)

    # harmônicos baseados em proporções de PI
    # concentrações relativas: PI(4,5)P2: 100, PI(3)P: 80, PI(4)P: 60, PI: 200
    pi_ratios = [100, 80, 60, 200]

    harmonics = sum((ratio/100.0) * np.sin(2*np.pi*(0.1*(i+2))*t) for i, ratio in enumerate(pi_ratios))

    signal = fundamental + 0.5 * harmonics

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal

if __name__ == "__main__":
    pulse = generate_life_pulse()
    print(f"Generated life pulse with {len(pulse)} samples.")
    print(f"Max amplitude: {np.max(pulse):.2f}")
