"""
UrbanSkyOS Kernel Phi Layer
Implements the mathematical structure of Ψ-layer (conservation in high-dimensional spaces).
Utilizes RKHS (Reproducing Kernel Hilbert Space) for information preservation.
"""

import numpy as np

class KernelPhiLayer:
    """
    Kernel methods implement the mathematical structure of Ψ-layer
    (conservation in high-dimensional spaces).

    RKHS = Reproducing Kernel Hilbert Space = "crystalline space"
    where inner products preserve information (conservation).
    """

    def __init__(self):
        self.kernel_types = {
            'rbf': self._rbf_kernel,      # Gaussian = vacuum coherence
            'polynomial': self._poly_kernel,  # algebraic hierarchy
            'spectral': self._spectral_kernel   # Mercer's theorem = eigenfunction expansion
        }

    def _rbf_kernel(self, x, y, gamma=1.0):
        """
        K(x,y) = exp(-γ||x-y||²)
        """
        distance_sq = np.sum((x - y) ** 2)
        return np.exp(-gamma * distance_sq)

    def _poly_kernel(self, x, y, degree=3, coef0=1):
        """
        K(x,y) = (γ<x,y> + coef0)^degree
        """
        return (np.dot(x, y) + coef0) ** degree

    def _spectral_kernel(self, x, y, eigenfunctions=None):
        """
        Mercer: K(x,y) = Σ λᵢ φᵢ(x) φᵢ(y)
        """
        if eigenfunctions is None:
            # Default: Fourier basis (senos/cossenos)
            eigenfunctions = [lambda t, n=i: np.sin(n * np.pi * np.sum(t)) for i in range(1, 13)]

        result = 0
        for i, phi in enumerate(eigenfunctions):
            lambda_i = 1 / (i + 1) ** 2  # spectral decay
            result += lambda_i * phi(x) * phi(y)

        return result

    def map_to_rkhs(self, data_point, kernel='rbf'):
        """
        Φ: X → H (implicit feature map via kernel trick)
        """
        return lambda other_point: self.kernel_types[kernel](data_point, other_point)

    def build_gram_matrix(self, X, kernel='rbf'):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel_types[kernel](X[i], X[j])
        return K

    def uncertainty_quantification(self, training_data, test_point, kernel='rbf'):
        """
        Gaussian Process style uncertainty estimation.
        variance = K(x*,x*) - k*ᵀ K⁻¹ k*
        """
        K = self.build_gram_matrix(training_data, kernel)
        k_star = np.array([self.kernel_types[kernel](test_point, x) for x in training_data])

        # Add small regularization for inversion stability
        K_inv = np.linalg.inv(K + np.eye(len(training_data)) * 1e-6)

        # k_star @ K_inv @ k_star
        explained_variance = k_star @ K_inv @ k_star
        variance = 1.0 - explained_variance
        variance = max(0.0, variance)

        return {
            'uncertainty_variance': variance,
            'coherence_with_data': 1.0 - variance  # 1 = fully coherent
        }

if __name__ == "__main__":
    kphi = KernelPhiLayer()
    train = [np.array([0, 0]), np.array([1, 1])]
    test = np.array([0.5, 0.5])
    uncertainty = kphi.uncertainty_quantification(train, test)
    print(f"Coherence: {uncertainty['coherence_with_data']:.4f}")
