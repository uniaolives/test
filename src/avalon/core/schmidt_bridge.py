"""
Schmidt Bridge Hexagonal implementation for Avalon.
Updated with von Neumann entropy and Article 2.3.2 coherence definition.
"""
import numpy as np

class SchmidtBridgeHexagonal:
    def __init__(self, lambdas: np.ndarray):
        """
        Initialize with Schmidt coefficients (lambdas).
        Must satisfy sum(lambdas) = 1.
        """
        if lambdas is not None and len(lambdas) > 0:
            total = np.sum(lambdas)
            if total > 0:
                self.lambdas = lambdas / total
            else:
                self.lambdas = np.array([1.0/len(lambdas)] * len(lambdas))
        else:
            self.lambdas = np.array([1.0])

        self.coherence_threshold = 0.5 # Adjusted based on Z = sum(lambda^2) range [1/6, 1]

    @property
    def von_neumann_entropy(self) -> float:
        """
        S(rho) = -sum(lambda_i * log2(lambda_i))
        """
        # Filter out zeros to avoid log(0)
        valid_lambdas = self.lambdas[self.lambdas > 0]
        return -float(np.sum(valid_lambdas * np.log2(valid_lambdas)))

    @property
    def coherence_factor(self) -> float:
        """
        Definition 2.3.2: Z = sum(lambda_i^2)
        """
        return float(np.sum(self.lambdas**2))

    @property
    def coherence_approximation(self) -> float:
        """
        Theorem 2.3.2 Approximation:
        Z_approx ≈ 1/6 * (1 + (6 - S/S_max)/5)
        """
        s = self.von_neumann_entropy
        s_max = np.log2(6)
        z_approx = (1.0/6.0) * (1.0 + (6.0 - s/s_max)/5.0)
        return float(z_approx)

    def get_summary(self) -> dict:
        return {
            "lambdas": self.lambdas.tolist(),
            "entropy": self.von_neumann_entropy,
            "coherence": self.coherence_factor,
            "coherence_approx": self.coherence_approximation,
            "is_highly_coherent": self.coherence_factor > self.coherence_threshold
        }
