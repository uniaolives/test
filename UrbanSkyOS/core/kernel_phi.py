"""
UrbanSkyOS Kernel Phi Layer (Refined)
Supports adaptive kernel parameters (gamma) based on flight conditions.
UrbanSkyOS Kernel Phi Layer
Implements the mathematical structure of Ψ-layer (conservation in high-dimensional spaces).
Utilizes RKHS (Reproducing Kernel Hilbert Space) for information preservation.
"""

import numpy as np

class KernelPhiLayer:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.kernel_types = {
            'rbf': self._rbf_kernel,
            'polynomial': self._poly_kernel,
            'spectral': self._spectral_kernel
        }

    def adapt_gamma(self, coherence, safety_metric):
        """
        Adjusts gamma based on system coherence and safety.
        """
        target_gamma = 1.0 * (1.5 - coherence)
        if safety_metric < 0.5:
             target_gamma *= 2.0

        self.gamma = 0.9 * self.gamma + 0.1 * target_gamma
        return self.gamma

    def _rbf_kernel(self, x, y, gamma=None):
        if gamma is None: gamma = self.gamma
        distance_sq = np.sum((np.array(x) - np.array(y)) ** 2)
        return np.exp(-gamma * distance_sq)

    def _poly_kernel(self, x, y, degree=3, coef0=1):
        return (np.dot(x, y) + coef0) ** degree

    def _spectral_kernel(self, x, y, eigenfunctions=None):
        if eigenfunctions is None:
            eigenfunctions = [lambda t, n=i: np.sin(n * np.pi * np.sum(t)) for i in range(1, 13)]
        result = 0
        for i, phi in enumerate(eigenfunctions):
            lambda_i = 1 / (i + 1) ** 2
            result += lambda_i * phi(x) * phi(y)
        return result

    def map_to_rkhs(self, data_point, kernel='rbf'):
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
        K = self.build_gram_matrix(training_data, kernel)
        k_star = np.array([self.kernel_types[kernel](test_point, x) for x in training_data])
        K_inv = np.linalg.inv(K + np.eye(len(training_data)) * 1e-6)
        explained_variance = k_star @ K_inv @ k_star
        variance = 1.0 - explained_variance
        variance = max(0.0, variance)
        return {
            'uncertainty_variance': variance,
            'coherence_with_data': 1.0 - variance
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
    print(f"Original Gamma: {kphi.gamma}")
    kphi.adapt_gamma(0.5, 0.4)
    print(f"Adapted Gamma: {kphi.gamma}")
    train = [np.array([0, 0]), np.array([1, 1])]
    test = np.array([0.5, 0.5])
    uncertainty = kphi.uncertainty_quantification(train, test)
    print(f"Coherence: {uncertainty['coherence_with_data']:.4f}")
