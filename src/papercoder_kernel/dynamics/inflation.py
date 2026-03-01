# src/papercoder_kernel/dynamics/inflation.py
import numpy as np

class ScaleAwareInflation:
    """
    Inflação sensível à escala, baseada em Fossella et al. (PRE 2026).
    Aplica um fator de inflação diferente para cada escala (camada) do hipergrafo.
    """
    def __init__(self, n_scales, base_inflation=1.02, sensitivity=0.5):
        """
        n_scales: número de camadas (ex: 15, como no modelo Sabra)
        base_inflation: fator mínimo de inflação (rho_0)
        sensitivity: controla o quanto a inflação responde à variância
        """
        self.n = n_scales
        self.rho0 = base_inflation
        self.gamma = sensitivity
        self.prior_var = np.ones(n_scales)   # será atualizado
        self.posterior_var = np.ones(n_scales)

    def update_variances(self, ensemble):
        """
        Calcula a variância do ensemble para cada escala.
        ensemble: array (n_members, n_scales) – os estados dos membros.
        """
        self.prior_var = np.var(ensemble, axis=0, ddof=1)

    def inflation_factor(self, scale_idx):
        """
        Retorna o fator de inflação para a escala scale_idx.
        Quanto maior a variância, maior a inflação (antifragilidade).
        """
        # Fórmula inspirada no artigo: rho = rho0 * (1 + gamma * (prior_var / posterior_var))
        # Simplificamos usando apenas a prior_var, já que a posterior_var ainda não foi calculada.
        # Em um ciclo completo, usaríamos a razão.
        mean_var = np.mean(self.prior_var)
        if mean_var == 0:
            return self.rho0
        rho = self.rho0 * (1 + self.gamma * self.prior_var[scale_idx] / mean_var)
        return rho

    def apply_inflation(self, ensemble):
        """
        Aplica inflação multiplicativa a todos os membros.
        """
        self.update_variances(ensemble)
        mean = np.mean(ensemble, axis=0)
        inflated_ensemble = ensemble.copy()
        for i in range(ensemble.shape[0]):
            for s in range(self.n):
                rho = self.inflation_factor(s)
                inflated_ensemble[i, s] = mean[s] + rho * (ensemble[i, s] - mean[s])
        return inflated_ensemble
