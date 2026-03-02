"""
Theory of Projection for Influence Functions and Effective Dimension.
Based on arXiv:2602.10449v1 (Hu et al.).
"""

import numpy as np
from typing import Tuple

def effective_dimension(F: np.ndarray, lambda_reg: float) -> Tuple[float, np.ndarray]:
    """
    Calcula a dimensão efetiva d_λ(F) = tr(F (F + λ I)^{-1}).

    Parâmetros:
        F : matriz simétrica positiva semidefinida (ex: Fisher Information Matrix).
        lambda_reg : parâmetro de regularização (escala de suavização λ).

    Retorna:
        d_eff : dimensão efetiva.
        contrib : vetor com as contribuições individuais de cada autovalor.
    """
    # Cálculo dos autovalores
    eigvals = np.linalg.eigvalsh(F)
    # Garantir autovalores não-negativos
    eigvals = np.maximum(eigvals, 0)

    # Cada contribuição é λ_i / (λ_i + λ)
    contrib = eigvals / (eigvals + lambda_reg)
    d_eff = np.sum(contrib)

    return d_eff, contrib

def random_projection_cost(d_eff: float, epsilon: float) -> int:
    """
    Calcula o custo de projeção (m) necessário para preservar a coerência.
    m ∝ d_λ(F) / ε²
    """
    return int(np.ceil(d_eff / (epsilon**2)))

if __name__ == "__main__":
    # Exemplo de uso com uma matriz simulada
    np.random.seed(42)
    d = 100
    r = 20   # rank real
    U = np.random.randn(d, r)
    # autovalores decaindo
    eigs = np.exp(-0.5 * np.arange(r))
    F = U @ np.diag(eigs) @ U.T

    lambda_reg = 0.1
    d_eff, contrib = effective_dimension(F, lambda_reg)

    print(f"Rank real da matriz: {r}")
    print(f"Dimensão Efetiva d_λ(F) na escala λ={lambda_reg}: {d_eff:.4f}")
    print(f"Custo de Projeção estimado (ε=0.1): {random_projection_cost(d_eff, 0.1)}")
