# src/papercoder_kernel/dynamics/repair.py
import numpy as np
from .inflation import ScaleAwareInflation

class MRN_RepairComplex:
    """
    Análogo ao complexo MRE11-RAD50-NBS1 para detecção e reparo de quebras
    na estrutura de dados do Disco de Festo.
    """
    def __init__(self, ensemble, inflation_module):
        self.ensemble = ensemble          # membros do EnKF
        self.inflation = inflation_module # inflação sensível à escala
        self.repair_log = []

    def detect_breaks(self, coherence_threshold=0.3):
        """
        Identifica regiões da espiral onde a coerência entre membros do ensemble
        cai abaixo do limiar – análogo a quebras de dupla fita no DNA.
        Retorna lista de índices (posições na espiral) que precisam de reparo.
        """
        # Coerência é medida como 1 - variância normalizada
        var = np.var(self.ensemble, axis=0)
        max_var = np.max(var)
        if max_var == 0:
            return np.array([])
        coherence = 1 - var / max_var
        breaks = np.where(coherence < coherence_threshold)[0]
        return breaks

    def recruit_repair(self, break_indices):
        """
        Para cada quebra, aplica uma rodada extra de assimilação com inflação
        localizada, forçando os membros a convergirem para uma solução consistente.
        """
        for idx in break_indices:
            # Extrai a região ao redor da quebra (janela de 5 posições)
            start = max(0, idx-2)
            stop = min(self.ensemble.shape[1], idx+3)
            window = slice(start, stop)

            # Calcula a média atual na janela
            mean_window = np.mean(self.ensemble[:, window], axis=0)

            # Aplica inflação extra na janela (fator 2x maior)
            for member in range(self.ensemble.shape[0]):
                for pos in range(start, stop):
                    # For performance in large ensembles, we check if prior_var is updated
                    # inflation_factor computes rho using prior_var
                    rho = self.inflation.inflation_factor(pos) * 2.0
                    self.ensemble[member, pos] = mean_window[pos - start] + \
                                                   rho * (self.ensemble[member, pos] - mean_window[pos - start])

            self.repair_log.append(idx)

    def verify_suture(self, known_fragments):
        """
        Compara as regiões reparadas com fragmentos conhecidos (split-GFP).
        Se a diferença for pequena, a sutura é considerada bem‑sucedida.
        """
        # known_fragments: dicionário {posição: valor real}
        if not known_fragments:
            return True

        errors = []
        for pos, real_val in known_fragments.items():
            estimated = np.mean(self.ensemble[:, pos])
            errors.append(abs(estimated - real_val))
        mean_error = np.mean(errors)
        return mean_error < 0.1  # limiar arbitrário
