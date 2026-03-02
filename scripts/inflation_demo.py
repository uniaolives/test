# scripts/inflation_demo.py
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.dynamics.inflation import ScaleAwareInflation

def main():
    print("--- Scale Aware Inflation Demo (Fossella et al. 2026) ---")

    n_scales = 15  # Modelo Sabra
    n_members = 20

    # Gerar ensemble sintético com variância decrescente (comum em cascatas de energia)
    # k ~ 2^n
    # variância ~ k^(-2/3) (Kolmogorov)
    scales = np.arange(n_scales)
    variances = 2.0**(scales * -2/3)

    ensemble = np.random.normal(0, 1, (n_members, n_scales)) * np.sqrt(variances)

    print(f"Ensemble inicial gerado: {n_members} membros, {n_scales} escalas.")
    print(f"Variância média inicial: {np.mean(np.var(ensemble, axis=0)):.6f}")

    # Inicializar inflação
    sai = ScaleAwareInflation(n_scales, base_inflation=1.01, sensitivity=0.8)

    # Aplicar inflação
    inflated_ensemble = sai.apply_inflation(ensemble)

    print("\nFatores de inflação por escala:")
    for s in range(n_scales):
        rho = sai.inflation_factor(s)
        print(f"  Escala {s:2d}: rho = {rho:.4f} (Var = {sai.prior_var[s]:.6f})")

    print(f"\nVariância média após inflação: {np.mean(np.var(inflated_ensemble, axis=0)):.6f}")

    # Verificar se as escalas com maior variância relativa sofreram maior inflação
    # (No nosso caso, as primeiras escalas têm maior variância absoluta,
    # mas a fórmula usa a variância relativa à média das variâncias).

if __name__ == "__main__":
    main()
