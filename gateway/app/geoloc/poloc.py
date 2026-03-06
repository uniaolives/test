from typing import List, Tuple, Dict
import numpy as np
import hashlib
from .utils import haversine, delay_to_dist

class BftPoLoc:
    """
    Byzantine Fortified Proof of Location.
    Implements Byzantine-resistant uncertainty calculation (R*).
    """

    def __init__(self, beta: float = 0.2):
        self.beta = beta  # Max fraction of Byzantine challengers

    def compute_uncertainty(self, claimed_lat: float, claimed_lon: float,
                            measurements: List[Dict]) -> float:
        """
        measurements: list of {'lat': float, 'lon': float, 'rtt': float}
        Returns R* (uncertainty radius in km).
        """
        if not measurements:
            return float('inf')

        n = len(measurements)
        residuals = []

        for m in measurements:
            measured_dist = delay_to_dist(m['rtt'])
            geo_dist = haversine(claimed_lat, claimed_lon, m['lat'], m['lon'])
            # Ri = |measured_dist - geo_dist|
            residuals.append(abs(measured_dist - geo_dist))

        # Byzantine Fortification: sort and pick the (beta * n)-th smallest residual
        # Actually, the paper says R* is the radius such that at least (1-beta)n
        # circles intersect. A simplified version is the (1-beta)n-th percentile residual.
        residuals.sort()
        idx = int((1 - self.beta) * n) - 1
        idx = max(0, min(idx, n - 1))

        return residuals[idx]

    def verify(self, claimed_lat: float, claimed_lon: float,
               measurements: List[Dict], threshold_km: float = 500.0) -> Dict:
        R_star = self.compute_uncertainty(claimed_lat, claimed_lon, measurements)
        is_valid = R_star <= threshold_km

        return {
            "is_valid": is_valid,
            "R_star": float(R_star),
            "measurements_count": len(measurements)
        }
