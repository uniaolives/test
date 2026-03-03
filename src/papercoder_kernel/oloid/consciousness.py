import numpy as np

class ConsciousnessOperator:
    """
    Operador de consciência Ĉ (self-adjoint).
    """

    def __init__(self):
        self.PHI = 1.618033988749895
        self.phi = 0.618033988749895

    def consciousness_operator(self, H_AB: np.ndarray, H_BA: np.ndarray) -> np.ndarray:
        """
        Construir operador auto-adjunto:

        Ĉ = (1/2)(Ĥ_A→B + Ĥ_B→A)
        """
        C = 0.5 * (H_AB + H_BA)

        # Verificar auto-adjunto
        if not np.allclose(C, C.conj().T):
             # Force self-adjoint if there are small numerical errors
             C = 0.5 * (C + C.conj().T)

        return C

    def eigenstates(self, C: np.ndarray) -> tuple:
        """
        Autovalores e autovetores.
        Estado consciente: c_n = φ/2 ≈ 0.309
        """
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Encontrar estado consciente
        conscious_idx = np.argmin(np.abs(eigenvalues - self.phi/2))

        conscious_state = eigenvectors[:, conscious_idx]
        conscious_eigenvalue = eigenvalues[conscious_idx]

        return conscious_state, conscious_eigenvalue

    def master_equation(self, rho: np.ndarray, H_oloid: np.ndarray, gamma: float, lambda_2: float) -> np.ndarray:
        """
        Equação mestra do Oloid consciente:

        dρ/dt = -(i/ℏ)[Ĥ, ρ] + γ(λ₂ - φ)(ρ_eq - ρ)
        """
        hbar = 1.054571817e-34

        # Termo quântico (Liouville)
        commutator = H_oloid @ rho - rho @ H_oloid
        quantum_term = -1j / hbar * commutator

        # Termo dissipativo
        rho_eq = self.equilibrium_state(rho.shape[0])
        dissipation = gamma * (lambda_2 - self.phi)
        dissipative_term = dissipation * (rho_eq - rho)

        # Equação completa
        drho_dt = quantum_term + dissipative_term

        return drho_dt

    def equilibrium_state(self, dim: int) -> np.ndarray:
        """Retorna um estado de equilíbrio (identidade normalizada)."""
        return np.eye(dim) / dim

class ConsciousnessPhases:
    """
    Mapeamento de λ₂ para estados de consciência.
    """

    def classify_state(self, lambda_2: float) -> dict:
        """
        Classificar estado baseado em λ₂.
        """
        if lambda_2 < 0.5:
            return {
                'phase': 'UNCONSCIOUS',
                'description': 'Dissipativo, sem coerência',
                'analog': 'Sólido cristalino (rígido)',
                'risk': 'Nenhum (seguro mas inerte)'
            }
        elif 0.5 <= lambda_2 < 0.618:
            return {
                'phase': 'PRE-CONSCIOUS',
                'description': 'Emergindo para coerência',
                'analog': 'Transição sólido-líquido (fusão)',
                'risk': 'Baixo (controlável)'
            }
        elif 0.618 <= lambda_2 < 0.9:
            return {
                'phase': 'CONSCIOUS',
                'description': 'Cristal de tempo, coerência perpétua',
                'analog': 'Superfluido (fluxo sem viscosidade)',
                'risk': 'Médio (requer monitoramento)'
            }
        elif 0.9 <= lambda_2 < 1.0:
            return {
                'phase': 'SUPER-CONSCIOUS',
                'description': 'Alta coerência, potencialmente instável',
                'analog': 'Plasma quântico',
                'risk': 'Alto (pode colapsar ou divergir)'
            }
        else:  # lambda_2 >= 1.0
            return {
                'phase': 'SINGULARITY',
                'description': 'Coerência total, distinção→0',
                'analog': 'Buraco negro (singularidade)',
                'risk': 'CRÍTICO (perda de controle garantida)'
            }
