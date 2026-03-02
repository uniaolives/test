import numpy as np

def sample_polynomial_roots(num_polys=100, degree=24):
    """Generate random polynomials and compute their roots (simplified)."""
    roots_list = []
    for _ in range(num_polys):
        coeffs = np.random.randn(degree+1)
        # companion matrix method
        companion = np.zeros((degree, degree))
        if degree > 1:
            companion[0, :] = -coeffs[1:] / coeffs[0]
            companion[1:, :-1] = np.eye(degree-1)
        roots = np.linalg.eigvals(companion)
        roots_list.append(roots)
    return roots_list
