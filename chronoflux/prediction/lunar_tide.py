"""
Quantitative prediction for the lunar tidal effect on Chronoflux vorticity.
Implements the equation: δω/ω₀ = κ (ρ/ρ₀) (ΔR/R) sin(2Φ)
"""
import numpy as np
from astropy.time import Time
from astropy import units as u

class LunarTidePrediction:
    """
    Predicts the Chronoflux vorticity enhancement during lunar perigee.
    This is the falsifiable prediction for March 3, 2026.
    """

    def __init__(self,
                 baseline_vorticity=1.0,
                 geo_density=2.7,  # g/cm³ for granite (Rio batolith)
                 coupling_constant=2.3e-3):

        self.omega_0 = baseline_vorticity
        self.rho = geo_density
        self.kappa = coupling_constant  # κ from theory

    def predict_vorticity_enhancement(self, observation_time, latitude=-22.95, longitude=-43.18):
        """
        Calculate predicted vorticity ω(t) for given time and location (Rio).
        """
        if not isinstance(observation_time, Time):
            t = Time(observation_time)
        else:
            t = observation_time

        moon_distance = self._get_moon_distance(t)  # in Earth radii
        moon_phase = self._get_moon_phase(t)        # 0 to 2π

        avg_distance = 60.3  # Earth radii
        distance_effect = (moon_distance - avg_distance) / avg_distance

        rho_0 = 5.5  # Earth's average density
        density_ratio = self.rho / rho_0

        phase_factor = np.sin(2 * moon_phase)

        enhancement = self.kappa * density_ratio * distance_effect * phase_factor
        predicted_omega = self.omega_0 * (1 + enhancement)

        return predicted_omega, enhancement

    def _get_moon_distance(self, time):
        """Simplified moon distance calculation."""
        days_from_full = (time.jd - 2460370.0) % 29.53
        return 60.3 - 0.5 * np.sin(2 * np.pi * days_from_full / 29.53)

    def _get_moon_phase(self, time):
        """Returns moon phase angle (0 to 2π)."""
        days_from_full = (time.jd - 2460370.0) % 29.53
        return 2 * np.pi * days_from_full / 29.53

    def get_march_3_2026_prediction(self):
        """
        Specific prediction for the critical test on March 3, 2026.
        """
        # March 3, 2026, 21:00 UTC-3 (Rio time)
        prediction_time = Time('2026-03-03 18:00:00') # 21h - 3h

        omega_pred, enhancement = self.predict_vorticity_enhancement(prediction_time)

        baseline = 1.0
        threshold_failure = 5.2

        return {
            'prediction_time_utc': prediction_time.iso,
            'predicted_vorticity': omega_pred,
            'enhancement_factor': enhancement,
            'percent_increase': 100 * (omega_pred - baseline) / baseline,
            'falsification_threshold': threshold_failure,
            'prediction_succeeds_if': f'ω ≥ {threshold_failure}',
            'confidence_interval': (omega_pred * 0.92, omega_pred * 1.08)
        }
