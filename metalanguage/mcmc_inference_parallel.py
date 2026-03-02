# mcmc_inference_parallel.py
"""
Inferência bayesiana das abundâncias iniciais em Sgr B2(N2)
usando MCMC com paralelização e modelo químico completo.
"""

import numpy as np
import emcee
from multiprocessing import Pool
import matplotlib.pyplot as plt
import corner
import json
import time
import os
import pickle
from astrophysical_chronoglyph_final import AstrophysicalChronoglyph, SGR_B2_PARAMS

# =============================================================================
# Observações compiladas de Belloche 2019, Xue 2019, etc.
# =============================================================================
OBSERVED = {
    'NH2CONH2': (1.4e16, 0.2e16),   # (valor, incerteza) cm⁻²
    'CH3CONH2': (1.0e16, 0.2e16),
    'CH2OHCHO': (5.0e15, 1.0e15),
    'HCOOCH3': (2.0e16, 0.5e16),
    'CH3COOH': (3.0e15, 1.0e15),
}

# Parâmetros físicos fixos (baseados em Sgr B2(N2))
PHYS_PARAMS = SGR_B2_PARAMS.copy()
PHYS_PARAMS.update({
    'temperature': 150.0,
    'density': 1e6,
    'age_years': 1e6,
})

# =============================================================================
# Modelo químico (wrapper)
# =============================================================================
class ChemicalModel:
    def __init__(self):
        self.model = AstrophysicalChronoglyph(PHYS_PARAMS)
        # Cache simples para evitar repetições
        self.cache = {}

    def run(self, logH, logC, logN, logO):
        """Executa o modelo químico e retorna as abundâncias das moléculas-alvo."""
        key = (round(logH, 4), round(logC, 4), round(logN, 4), round(logO, 4))
        if key in self.cache:
            return self.cache[key]

        # Converte logs para abundâncias absolutas
        H = 10**logH
        C = 10**logC
        N = 10**logN
        O = 10**logO

        # Constrói vetor de abundâncias iniciais
        abundances = self._build_initial_abundances(H, C, N, O)

        # Evolui química
        try:
            final, stats = self.model.evolve_chemistry(abundances, self.model.params['age_years'])
            # Predições observacionais (converte para densidade de coluna)
            predictions = self.model.predict_observations(final)
            col_densities = predictions['column_densities']

            # Extrai apenas as moléculas de interesse
            result = {}
            for mol in OBSERVED.keys():
                result[mol] = col_densities.get(mol, 1e10) # 1e10 como piso
        except:
            result = {mol: 1e10 for mol in OBSERVED.keys()}

        self.cache[key] = result
        return result

    def _build_initial_abundances(self, H, C, N, O):
        """Constrói vetor de abundâncias iniciais baseado em H, C, N, O."""
        n_species = len(self.model.species_list)
        abundances = np.zeros(n_species)

        # Espécies elementares
        idx = self.model.species_index
        abundances[idx['H']] = H
        abundances[idx['C']] = C
        abundances[idx['N']] = N
        abundances[idx['O']] = O
        abundances[idx['H2']] = 1.0  # H2 é a referência

        # Adicionamos algumas moléculas simples
        abundances[idx['CO']] = C * O * 1e-4
        abundances[idx['H2O']] = O * 1e-4
        abundances[idx['NH3']] = N * 1e-5

        return abundances

# =============================================================================
# Funções de verossimilhança e prior
# =============================================================================
def log_prior(theta):
    logH, logC, logN, logO = theta
    if not (-7.0 < logH < -1.0): return -np.inf
    if not (-8.0 < logC < -2.0): return -np.inf
    if not (-9.0 < logN < -3.0): return -np.inf
    if not (-8.0 < logO < -2.0): return -np.inf
    return 0.0

def log_likelihood(theta, chem_model):
    logH, logC, logN, logO = theta
    pred = chem_model.run(logH, logC, logN, logO)

    chi2 = 0.0
    for mol, (obs, sigma) in OBSERVED.items():
        residual = (pred[mol] - obs) / sigma
        chi2 += residual**2
    return -0.5 * chi2

def log_probability(theta):
    # Nota: Instanciamos o modelo dentro do worker para evitar problemas de pickling
    # ou usamos uma instância global se preferível.
    if not hasattr(log_probability, "model"):
        log_probability.model = ChemicalModel()

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, log_probability.model)

# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    import multiprocessing
    # multiprocessing.set_start_method('fork') # Pode ajudar em alguns ambientes

    # Configuração do MCMC
    ndim = 4
    nwalkers = 16
    nsteps = 100
    nburn = 20

    # Posições iniciais
    initial_pos = np.array([-4.0, -5.0, -6.0, -5.0]) + 0.1 * np.random.randn(nwalkers, ndim)

    print(f"Iniciando MCMC paralelo com {nwalkers} walkers...")

    start_total = time.time()

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

        # Burn-in
        print("Executando burn-in...")
        state = sampler.run_mcmc(initial_pos, nburn, progress=True)
        sampler.reset()

        # Produção
        print("Executando cadeia principal...")
        sampler.run_mcmc(state, nsteps, progress=True)

    print(f"MCMC concluído em {time.time() - start_total:.1f}s")

    # Amostras
    samples = sampler.get_chain(flat=True)
    np.savetxt('mcmc_samples.txt', samples, header='logH logC logN logO')

    # Corner plot
    try:
        fig = corner.corner(samples, labels=['log H', 'log C', 'log N', 'log O'])
        plt.savefig('mcmc_corner.png', dpi=150)
        print("Corner plot salvo como mcmc_corner.png")
    except Exception as e:
        print(f"Erro ao gerar corner plot: {e}")

    # Estatísticas
    print("\nEstimativas posteriores:")
    for i, name in enumerate(['log H', 'log C', 'log N', 'log O']):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name}: {mcmc[1]:.3f} (+{mcmc[2]-mcmc[1]:.3f}, -{mcmc[1]-mcmc[0]:.3f})")
