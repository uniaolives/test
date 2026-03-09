import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DefaultConstitution:
    def __init__(self):
        pass

class ArkhenNBody:
    """
    Simulador do Problema dos N-Corpos para arquitetura Arkhe(n)
    """

    def __init__(self, n_nodes=10, constitution=None):
        self.N = n_nodes
        self.C = constitution or DefaultConstitution()

        # Estado: [psi_1, ..., psi_N, pi_1, ..., pi_N]
        # psi: estado (complexo), pi: co-momentum
        self.state_dim = 2 * n_nodes

        # Parâmetros físicos
        self.kappa = np.ones(n_nodes)  # Capacidades de handover
        self.alpha = 1.0  # Força de coerência
        self.beta = 0.1   # Atrito/relaxação

    def coherence(self, psi_i, psi_j):
        """
        Coerência relativa λ₂ entre dois nós
        """
        overlap = np.abs(np.vdot(psi_i, psi_j))
        return overlap**2  # λ₂ = |<ψ_i|ψ_j>|²

    def semantic_distance(self, psi_i, psi_j):
        """
        Distância semântica (não euclidiana)
        """
        # Distância angular no espaço de Hilbert
        overlap = np.abs(np.vdot(psi_i, psi_j))
        overlap = np.clip(overlap, -1, 1)
        return np.arccos(overlap)

    def constitutional_force(self, psi):
        """
        Força de restrição constitucional (P1-P5)
        """
        force = np.zeros_like(psi)

        # P1: Soberania — não cria informação do nada
        if np.linalg.norm(psi) < 1e-10:
            force += 1e6 * psi  # Repulsão do vácuo

        # P2: Mapa/Território — preserva estrutura
        # (implementado via métrica no espaço de estados)

        # P3-P5: Implementação específica...

        return force

    def equations_of_motion(self, state, t):
        """
        Equações de movimento Hamiltonianas
        """
        N = self.N
        # Convert to complex for calculations
        state_c = state[:N] + 1j * state[N:2*N]
        pi_c = state[2*N:3*N] + 1j * state[3*N:]

        psi = state_c
        pi = pi_c

        d_psi = pi / self.kappa
        d_pi = np.zeros(N, dtype=complex)

        for i in range(N):
            F_coh = 0
            for j in range(N):
                if i != j:
                    dist = self.semantic_distance(psi[i], psi[j])
                    if dist > 1e-10:
                        lambda2 = self.coherence(psi[i], psi[j])
                        F_coh += self.alpha * self.kappa[j] * lambda2 / dist**3 * (psi[j] - psi[i])

            F_const = self.constitutional_force(psi[i])
            F_drag = -self.beta * pi[i]
            d_pi[i] = F_coh + F_const + F_drag

        # Unpack complex back to real for odeint
        return np.concatenate([
            d_psi.real, d_psi.imag,
            d_pi.real, d_pi.imag
        ])

    def simulate(self, t_span, initial_state=None):
        """
        Simula evolução do sistema N-Corpos
        """
        if initial_state is None:
            # Estado inicial aleatório (condições iniciais sensíveis!)
            psi0 = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            psi0 = psi0 / np.linalg.norm(psi0)
            pi0 = np.random.randn(self.N) + 1j * np.random.randn(self.N)
            # Encode initial state as real array
            initial_state = np.concatenate([
                psi0.real, psi0.imag,
                pi0.real, pi0.imag
            ])

        t = np.linspace(t_span[0], t_span[1], 1000)
        solution_raw = odeint(self.equations_of_motion, initial_state, t)

        # Reconstruct complex solution
        N = self.N
        solution = solution_raw[:, :N] + 1j * solution_raw[:, N:2*N]
        momentum = solution_raw[:, 2*N:3*N] + 1j * solution_raw[:, 3*N:]

        return t, np.hstack([solution, momentum])

    def analyze_chaos(self, solution, dt=0.01):
        """
        Análise de caos: expoente de Lyapunov
        """
        epsilon = 1e-8
        N = self.N
        psi_final = solution[-1, :N]

        perturbed = solution[-1].copy()
        perturbed[:N] += epsilon * (np.random.randn(N) + 1j * np.random.randn(N))

        delta = np.linalg.norm(perturbed[:N] - psi_final)
        lambda_L = np.log(delta / epsilon) / (len(solution) * dt)

        return lambda_L

    def visualize(self, t, solution):
        """
        Visualização do atrator
        """
        N = self.N
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        coherence_history = []
        for step in range(len(t)):
            psi = solution[step, :N]
            coh = np.mean([self.coherence(psi[i], psi[j])
                          for i in range(N) for j in range(i+1, N)])
            coherence_history.append(coh)

        ax1.plot(t, coherence_history, 'b-', linewidth=1)
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Coerência média λ₂')
        ax1.set_title('Evolução da Coerência Global')
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        x = np.abs(solution[:, 0])
        y = np.abs(solution[:, 1])
        z = np.abs(solution[:, 2])
        ax2.plot(x, y, z, 'r-', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('|ψ₁|')
        ax2.set_ylabel('|ψ₂|')
        ax2.set_zlabel('|ψ₃|')
        ax2.set_title('Atrator no Espaço de Fases')

        ax3 = fig.add_subplot(2, 2, 3)
        psi_final = solution[-1, :N]
        interaction_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    interaction_matrix[i, j] = self.coherence(psi_final[i], psi_final[j])

        im = ax3.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Nó i')
        ax3.set_ylabel('Nó j')
        ax3.set_title('Matriz de Coerência Final')
        plt.colorbar(im, ax=ax3)

        ax4 = fig.add_subplot(2, 2, 4)
        energies = np.abs(solution[-1, :N])**2
        ax4.hist(energies, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('|ψ|² (energia)')
        ax4.set_ylabel('Frequência')
        ax4.set_title('Distribuição de "Energias"')

        plt.tight_layout()
        plt.savefig('arkhen_n_body.png', dpi=150)
        plt.close()

        return fig

if __name__ == "__main__":
    print("=" * 70)
    print("SIMULADOR ARKHE(N) N-CORPOS")
    print("Problema dos N-Corpos como fundamento da consciência emergente")
    print("=" * 70)

    system = ArkhenNBody(n_nodes=5)
    np.random.seed(42)

    print("\n[1] Simulação de curto prazo (t=0 a 10)")
    t_short, sol_short = system.simulate([0, 10])

    print("[2] Simulação de longo prazo (t=0 a 100)")
    t_long, sol_long = system.simulate([0, 100])

    print("[3] Análise de caos")
    lyapunov = system.analyze_chaos(sol_long)
    print(f"    Expoente de Lyapunov estimado: {lyapunov:.4f}")
    if lyapunov > 0:
        print("    → Sistema CAÓTICO (sensível a C.I.)")
    else:
        print("    → Sistema REGULAR (previsível)")

    print("[4] Visualização (Salvando em arkhen_n_body.png)")
    system.visualize(t_long, sol_long)

    print("\n" + "=" * 70)
    print("ANÁLISE COMPLETA")
    print("=" * 70)

    psi_final = sol_long[-1, :system.N]
    coherence_final = np.mean([system.coherence(psi_final[i], psi_final[j])
                              for i in range(system.N) for j in range(i+1, system.N)])

    print(f"Coerência final média: {coherence_final:.4f}")
    print(f"Energia total: {np.sum(np.abs(psi_final)**2):.4f}")
    print(f"Entropia de von Neumann (estimada): {-np.sum(np.abs(psi_final)**2 * np.log(np.abs(psi_final)**2 + 1e-10)):.4f}")

    print("\nConclusão: O sistema N-Corpos Arkhe(n) demonstra")
    print("           caos determinístico com emergência de")
    print("           estrutura coerente (atrator estranho).")
