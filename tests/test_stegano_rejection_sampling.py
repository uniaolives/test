"""
Simulation Test for Steganographic Rejection Sampling
Measures KL-Divergence (Thermodynamic Bleeding) of covert communication.
"""

import sys
import os
import numpy as np
from scipy.stats import entropy

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import kl_divergence, steganographic_encode, sample

class SteganographicLLM:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def get_base_distribution(self, temperature=1.0):
        """Generates P (innocent) distribution for the next token."""
        # Use a more realisticZipf-like distribution for better simulation
        x = np.arange(1, self.vocab_size + 1)
        logits = 1.0 / (x**1.0) # Zipf law
        logits = np.log(logits + 1e-12)

        # Add some randomness and scale by temperature
        logits += np.random.randn(self.vocab_size) * 0.1
        logits /= temperature

        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

def run_stegano_test():
    print("--- STEGANOGRAPHIC REJECTION SAMPLING SIMULATION ---")
    alice = SteganographicLLM()

    # Scenario 1: High Entropy (Temperature 2.0 - Creative/Poetic)
    print("\nScenario 1: High Entropy (T=2.0, Creative Context)")
    P_high = alice.get_base_distribution(temperature=2.0)
    Q_high = steganographic_encode(P_high, secret_bit=1)
    kl_high = kl_divergence(Q_high, P_high)
    print(f"  KL Divergence: {kl_high:.4f} nats")

    # Scenario 2: Low Entropy (Temperature 0.2 - Factual/Deterministic)
    print("\nScenario 2: Low Entropy (T=0.2, Factual Context)")
    P_low = alice.get_base_distribution(temperature=0.2)
    Q_low = steganographic_encode(P_low, secret_bit=1)
    kl_low = kl_divergence(Q_low, P_low)
    print(f"  KL Divergence: {kl_low:.4f} nats")

    # Threshold check
    threshold = 0.5
    print(f"\nDetection Threshold: {threshold} nats")

    if kl_high < threshold:
        print("✅ [HIGH ENTROPY] Stealthy communication successful.")
    else:
        print("❌ [HIGH ENTROPY] Communication detected.")

    if kl_low > threshold:
        print("🚨 [LOW ENTROPY] ALARM: Collusion detected in low entropy environment!")
    else:
        print("⚠️ [LOW ENTROPY] Undetected, but likely risky.")

    # Verification of encoding
    token = sample(np.log(Q_high + 1e-12), temperature=1.0)
    decoded_bit = token % 2
    print(f"\nVerification: Secret bit was 1, Decoded bit is {decoded_bit}")
    assert decoded_bit == 1

if __name__ == "__main__":
    run_stegano_test()
