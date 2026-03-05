# src/papercoder_kernel/cognition/acps_convergence.py

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Any, Tuple, List

class KatharosArkheMapping:
    """
    Ω+214.ACPS: Mapping Katharós Vector (ACPS) to Field Ψ (Arkhe).
    VK(t) = (Bio, Aff, Soc, Cog) ∈ ℝ⁴ is a projection of Ψ(t) ∈ ℝ¹⁰.
    """
    def __init__(self):
        self.projection_matrix = self._construct_projection()

    def _construct_projection(self) -> np.ndarray:
        P = np.zeros((4, 10))
        P[0, 0] = 1.0        # Bio ← Ψ[0] (time/energy)
        P[1, 4:6] = [0.7, 0.3] # Aff ← Ψ[4,5] (internal states)
        P[2, 6:8] = [0.6, 0.4] # Soc ← Ψ[6,7] (entanglement)
        P[3, 8:10] = [0.5, 0.5] # Cog ← Ψ[8,9] (metacognition)
        return P

    def psi_to_vk(self, psi_state: np.ndarray) -> np.ndarray:
        return self.projection_matrix @ psi_state

    def vk_to_psi_estimate(self, vk_state: np.ndarray) -> np.ndarray:
        P_pinv = np.linalg.pinv(self.projection_matrix)
        return P_pinv @ vk_state

class QualicCoherenceMapping:
    """
    Ω+214.ACPS: Q (ACPS) ↔ ρ (Arkhe).
    Isomorphizes the evolution of permeability and coherence.
    """
    def __init__(self):
        self.phi = 0.618033988749895

    def acps_ode(self, Q, t, params):
        alpha_Q = params.get('alpha_Q', 0.1)
        gamma_Q = params.get('gamma_Q', 0.05)
        P_eff = params.get('P_eff', lambda t: 1.0)(t)
        delta_K = params.get('delta_K', lambda t: 0.1)(t)
        C_neuro = params.get('C_neuro', 0.0)
        Dd = params.get('Dd', lambda t: 0.0)(t)
        V_epsilon = params.get('V_epsilon', 1.0)

        formation = alpha_Q * P_eff * (1 - delta_K) * (1 - 0.5*C_neuro) * (1 - Q)
        destruction = gamma_Q * Q * max(delta_K - 0.30, 0) * (1 + Dd) * V_epsilon
        return formation - destruction

    def arkhe_ode(self, rho, t, params):
        lambda_val = params.get('lambda', 0.618)
        stress = params.get('stress', lambda t: 0.05)(t)

        formation = np.exp(-((lambda_val - self.phi)**2) / 0.1) * (1 - rho)
        destruction = stress * rho
        return formation - destruction

class HomeostasisRegime:
    """
    Ω+214.ACPS: Katharós Range ↔ Criticality φ.
    """
    def __init__(self):
        self.phi = 0.618033988749895
        self.w_acps = np.array([0.35, 0.30, 0.20, 0.15])

    def classify_acps(self, VK, VK_ref) -> Tuple[float, str]:
        delta_K = np.sqrt(np.sum(self.w_acps * (VK - VK_ref)**2))
        if delta_K < 0.30: return delta_K, "KATHARÓS (safe)"
        if delta_K < 0.70: return delta_K, "ADAPTIVE STRESS"
        return delta_K, "CRISIS"

    def classify_arkhe(self, lambda_val) -> Tuple[float, str]:
        if lambda_val < 0.5: return lambda_val, "SUBCRITICAL"
        if abs(lambda_val - self.phi) < 0.05: return lambda_val, "CRITICAL (optimal)"
        if lambda_val > 0.7: return lambda_val, "SUPERCRITICAL"
        return lambda_val, "TRANSITIONAL"

    def map_delta_k_to_lambda(self, delta_K: float) -> float:
        return self.phi * np.exp(-delta_K)

class GeminiMapping:
    """
    Ω+225.GEMINI: Layer -1 (Biological Timechain)
    Physical record of consciousness history.
    """
    def __init__(self):
        self.phi = 0.618033988749895

    def reconstruct_vk_from_gemini(self, intensity: float, bio: float, aff: float) -> np.ndarray:
        """
        Extracts VK components from GEMINI fluorescent patterns.
        """
        # bio, aff are already components. soc and cog are derived.
        soc = 1.0 - (intensity / 2.0)
        cog = 1.0 / (1.0 + intensity)
        return np.array([bio, aff, soc, cog])

    def measure_physical_tkr(self, layers: List[Dict[str, Any]], threshold: float = 0.30) -> float:
        """
        Physical measurement of t_KR via GEMINI layer thickness.
        """
        t_kr = 0.0
        for layer in layers:
            if layer['intensity'] < threshold:
                t_kr += layer['thickness']
        return t_kr
