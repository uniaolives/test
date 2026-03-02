"""
Quantum State Vector manipulation for QHTTP.
"""
import numpy as np

class QuantumStateVector:
    def __init__(self, amplitudes):
        self.amplitudes = np.array(amplitudes)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
