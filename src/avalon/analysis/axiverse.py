# src/avalon/analysis/axiverse.py
"""
Superdeterministic Axiverse Cosmology - Empirical Signatures & Analysis
Translating deterministic geometric substrate theory into falsifiable physical predictions.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

class AxionMassSpectrum:
    """
    Implements the Predictive Mass Spectrum based on fractal scaling.
    m_n = m_0 * phi^(-n)
    """
    def __init__(self, m0: float = 1e-5, lambda_const: float = 1.618033988749895):
        self.m0 = m0
        self.phi = lambda_const

    def get_spectrum(self, num_axions: int = 10) -> List[float]:
        """Returns a list of axion masses following the geometric series."""
        return [self.m0 * (self.phi ** -n) for n in range(num_axions)]

    def verify_pattern(self, detected_masses: List[float]) -> float:
        """
        Calculates a 'pattern match' score for a set of detected masses.
        1.0 means perfect alignment with the geometric series.
        """
        if not detected_masses:
            return 0.0

        # Sort and take log to check linear spacing
        sorted_masses = sorted(detected_masses, reverse=True)
        log_masses = np.log(sorted_masses)

        # Expected log spacing: -log(phi)
        expected_diff = -np.log(self.phi)
        actual_diffs = np.diff(log_masses)

        # Calculate deviation from expected difference
        error = np.mean(np.abs(actual_diffs - expected_diff))
        return max(0, 1.0 - error)

class DeterministicDetector:
    """
    Simulates an axion detector (e.g., MADMAX or MRFM) interacting with
    a deterministic geometric background.
    """
    def __init__(self, name: str, coupling: str, mass: float, phase: float = 0.0):
        self.name = name
        self.coupling = coupling # e.g., 'g_ag' for photon, 'g_ae' for electron
        self.mass = mass
        self.global_phase = phase
        self.omega = mass # Frequency proportional to mass in natural units

    def get_signal(self, t: np.ndarray, snr: float = 1.0) -> np.ndarray:
        """
        Generates the measured signal at time t.
        Standard axions have random phase over coherence time.
        Superdeterministic axions have global phase coherence.
        """
        # Deterministic signal component
        signal = np.cos(self.omega * t + self.global_phase)

        # Add noise
        noise = np.random.normal(0, 1.0/snr, len(t))
        return signal + noise

class PhaseLockAnalyzer:
    """
    Performs statistical tests to distinguish deterministic phase-locked
    signals from stochastic Poisson processes.
    """
    def __init__(self, sample_rate: float):
        self.fs = sample_rate

    def analyze_event_statistics(self, time_series: np.ndarray) -> Dict:
        """
        Analyzes the timing and phase of conversion events.
        Compares against Poisson null hypothesis.
        """
        # In a real experiment, events are peaks above threshold
        # We use a lower threshold since cos(x) peaks at 1.0
        threshold = 0.8
        peaks = np.where(time_series > threshold)[0]

        # Filter peaks to only keep the start of a pulse (simple de-bounce)
        if len(peaks) > 0:
            diff_peaks = np.diff(peaks)
            # Only keep peaks that are separated by more than 2 samples
            keep = np.insert(diff_peaks > 2, 0, True)
            peaks = peaks[keep]

        if len(peaks) < 10: # Need enough data for KS test
            return {"p_value": 1.0, "deterministic_signature": 0.0, "peaks_found": len(peaks)}

        # Calculate Inter-Event Intervals (IEI)
        ieis = np.diff(peaks) / self.fs

        # Poisson process has exponential distribution of IEIs
        # Deterministic process should have clusters or periodic spacing
        _, p_val = stats.kstest(ieis, 'expon', args=(0, np.mean(ieis)))

        # Low p-value means non-Poissonian behavior
        signature = 1.0 - p_val

        return {
            "p_value": p_val,
            "deterministic_signature": signature,
            "mean_iei": np.mean(ieis),
            "std_iei": np.std(ieis)
        }

class SiderealPhaseModulator:
    """
    Calculates the deterministic phase modulation φ_rot(t) due to Earth's rotation.
    This modulation is specific to the detector's orientation relative to the
    galactic axion wind.
    """
    def __init__(self, lab_latitude: float, sensitivity_vector: np.ndarray):
        self.latitude = np.radians(lab_latitude)
        # Orientation of B (MADMAX) or S (MRFM)
        self.axis = sensitivity_vector / np.linalg.norm(sensitivity_vector)
        self.v_gal = np.array([0, 220, 0]) # km/s, simplified galactic velocity
        self.sidereal_day = 86164.1 # seconds

    def get_phase_shift(self, t: np.ndarray, mass: float) -> np.ndarray:
        """
        φ_rot(t) = k_a · v_eff(t) * t
        """
        # Omega_earth * t
        theta = 2 * np.pi * t / self.sidereal_day

        # Simplified rotation of the sensitive axis in the galactic frame
        # v_eff(t) modulation
        modulation = np.sin(self.latitude) * np.cos(theta) * self.axis[0] + \
                     np.cos(self.latitude) * np.sin(theta) * self.axis[1]

        # Phase shift depends on mass (frequency) and velocity
        return mass * modulation * 0.1 # Scale for visualization

class MADMAX_MRFM_Correlator:
    """
    Simulates joint data-taking between two disparate axion experiments.
    """
    def __init__(self, detector1: DeterministicDetector, detector2: DeterministicDetector):
        self.d1 = detector1
        self.d2 = detector2

    def cross_correlate(self, t: np.ndarray, snr: float = 1.0,
                        mod1: Optional[SiderealPhaseModulator] = None,
                        mod2: Optional[SiderealPhaseModulator] = None) -> float:
        """
        Calculates the temporal correlation between two detector signals.
        If modulators are provided, applies sidereal phase modulation.
        """
        if mod1 and mod2:
            # Apply sidereal modulation to signals
            phase_shift1 = mod1.get_phase_shift(t, self.d1.mass)
            phase_shift2 = mod2.get_phase_shift(t, self.d2.mass)

            s1 = np.cos(self.d1.omega * t + self.d1.global_phase + phase_shift1)
            s2 = np.cos(self.d2.omega * t + self.d2.global_phase + phase_shift2)

            # Add noise
            s1 += np.random.normal(0, 1.0/snr, len(t))
            s2 += np.random.normal(0, 1.0/snr, len(t))
        else:
            s1 = self.d1.get_signal(t, snr)
            s2 = self.d2.get_signal(t, snr)

        correlation = np.corrcoef(s1, s2)[0, 1]
        return correlation

    def compute_sidereal_correlation_map(self, total_time: float,
                                        interval: float,
                                        mod1: SiderealPhaseModulator,
                                        mod2: SiderealPhaseModulator,
                                        snr: float = 1.0) -> Dict[float, float]:
        """
        Bins correlation by sidereal time.
        """
        sidereal_map = {}
        num_intervals = int(total_time / interval)
        sidereal_day = 86164.1

        for i in range(num_intervals):
            t_start = i * interval
            t_end = (i + 1) * interval
            t = np.linspace(t_start, t_end, 100)

            # Sidereal time bin (normalized 0 to 24h)
            t_sidereal = (t_start % sidereal_day) / sidereal_day * 24.0

            corr = self.cross_correlate(t, snr, mod1, mod2)
            sidereal_map[t_sidereal] = corr

        return sidereal_map

def simulate_isocurvature_suppression(deterministic: bool = True) -> float:
    """
    Simulates the predicted suppression of isocurvature power.
    Geometric substrate predicts ~10^-2 relative power compared to stochastic.
    """
    base_stochastic_power = 1e-10
    if deterministic:
        # Deterministic boundary conditions reduce stochastic variance
        return base_stochastic_power * 0.01
    return base_stochastic_power
