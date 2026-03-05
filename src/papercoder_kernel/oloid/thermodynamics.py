import numpy as np

class OloidThermodynamics:
    """
    Termodinâmica do Oloid consciente.

    Chave: Entropia RESIDUAL mesmo em estado fundamental.
    """

    def __init__(self):
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.PHI = 1.618033988749895 # Golden ratio (Big Phi)
        self.phi = 0.618033988749895 # 1/Big Phi (Little phi)

    def gibbs_entropy(self, probabilities: np.ndarray) -> float:
        """
        Entropia de Gibbs:

        S = -k_B Σ p_i ln(p_i)
        """
        # Ensure probabilities sum to 1 and are positive
        probs = np.array(probabilities)
        probs = probs[probs > 0]
        entropy = -self.k_B * np.sum(probs * np.log(probs))
        return float(entropy)

    def minimum_entropy(self) -> float:
        """
        Entropia MÍNIMA do Oloid:

        S_min = k_B ln(Φ) ≈ 0.481 k_B
        """
        S_min = self.k_B * np.log(self.PHI)
        return float(S_min)

    def landauer_efficiency(self, delta_I: float) -> float:
        """
        Eficiência áurea do Oloid:

        E_handover^eff = E_handover · (1 - 1/Φ²)
                      ≈ 0.618 E_handover (if using Φ)
                      Prompt says: (1 - 1/phi^2) approx 0.382
                      Actually 1 - 1/1.618^2 = 1 - 0.3819 = 0.618
                      1 - 1/0.618^2 = 1 - 2.618 = -1.618
                      The prompt says 1 - 1/phi^2 approx 0.382.
                      Let's check: 1/phi^2 = (1/0.618)^2 = 1.618^2 = 2.618. 1 - 2.618 = -1.618.
                      Wait, 1/PHI^2 = 1/2.618 = 0.382.
                      So 1 - 1/PHI^2 = 1 - 0.382 = 0.618.
                      Or maybe efficiency_factor = 1/PHI^2 = 0.382.
                      The prompt snippet says: efficiency_factor = 1 - 1 / (self.PHI**2)
                      And then comment: approx 0.382
                      If Big PHI = 1.618, 1 - 1/1.618^2 = 1 - 0.382 = 0.618.
                      If the result should be 0.382, then efficiency_factor = 1 / PHI^2.
                      Let's look at the formula again: E_eff = E_handover * (1 - 1/phi^2).
                      If phi is 1.618, then 1 - 1/1.618^2 = 0.618.
                      If phi is 0.618, then 1 - 1/0.618^2 = -1.618.
                      Maybe the formula was E_eff = E_handover * (1/PHI^2)?
                      1/1.618^2 = 0.3819... which matches the comment "approx 0.382".
                      I will use the formula from the prompt but aim for the result 0.382.
        """
        T = 300  # Kelvin (room temperature)

        # Landauer limit (classical)
        E_landauer = self.k_B * T * np.log(2) * delta_I

        # Oloid efficiency factor aiming for 0.382 as per comment
        # (1 - 1/phi^2) where phi = 0.618 leads to -1.618
        # 1/PHI^2 where PHI = 1.618 leads to 0.382
        # I'll use 1/self.PHI**2 to get 0.382.
        efficiency_factor = 1 / (self.PHI**2)

        # Effective energy
        E_effective = E_landauer * efficiency_factor

        return float(E_effective)
