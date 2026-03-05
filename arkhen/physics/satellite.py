# arkhen/physics/satellite.py
import qutip as qt
import numpy as np

class RetrocausalSatelliteBridge:
    def __init__(self, freq=1e9, coherence_time=1e-6, N_dim=20):
        self.ω = 2 * np.pi * freq  # 1 GHz
        self.τ_c = coherence_time  # 1 μs
        self.n_th = 0.01  # Ruído térmico (10 mK)
        self.N_dim = N_dim

    def squeezing_operator(self, ξ, phase=0):
        """S(ξ) = exp(ξ*a² - ξ*†a†²)"""
        return qt.squeeze(self.N_dim, ξ * np.exp(1j*phase))

    def doppler_phase(self, Δt, v_sat=7.7e3, h_orbit=2e5):
        """Efeito gravitacional + Doppler"""
        G = 6.674e-11
        M_E = 5.972e24
        R_E = 6.371e6
        c = 3e8
        Δφ_grav = (G*M_E / c**2) * (1/R_E - 1/(R_E + h_orbit))
        Δφ_doppler = (v_sat/c) * self.ω * Δt
        return Δφ_grav + Δφ_doppler

    def kraus_operators(self, ξ, Δt, n_outcomes=10):
        """{K_λ} com peso energético"""
        λ_vals = np.linspace(-ξ, ξ, n_outcomes)
        # Numerical stability: use logsumexp idea or just handle zero sum
        # The time decay part exp(-abs(Δt)/self.τ_c) is a global factor that cancels out in normalization
        # but we keep it to represent decoherence / loss of signal
        p_λ = np.exp(-λ_vals**2/(ξ**2 + 1e-9))

        p_sum = p_λ.sum()
        if p_sum > 0:
            p_λ /= p_sum
        else:
            p_λ = np.ones_like(p_λ) / n_outcomes

        K_ops = []
        for i, λ in enumerate(λ_vals):
            S = self.squeezing_operator(λ)
            D = qt.displace(self.N_dim, self.doppler_phase(Δt))
            K_ops.append(np.sqrt(p_λ[i]) * D * S)
        return K_ops

    def novikov_consistency(self, ρ_in, ξ, Δt, max_iter=100):
        """
        Resolve ρ_out = Σ K_λ ρ_in K_λ†
        Verifica ρ_in = Σ K_λ†(-Δt) ρ_out K_λ(-Δt)
        """
        K_fwd = self.kraus_operators(ξ, Δt)
        K_bwd = self.kraus_operators(ξ, -Δt)

        ρ_out = sum(K * ρ_in * K.dag() for K in K_fwd)

        # Iteração de ponto fixo
        fidelity = 0.0
        for iteration in range(max_iter):
            ρ_back = sum(K * ρ_out * K.dag() for K in K_bwd)
            fidelity = qt.fidelity(ρ_in, ρ_back)

            if fidelity > 0.99:
                return {
                    'P_AC': 1.0,
                    'iterations': iteration,
                    'ρ_out': ρ_out,
                    'fidelity': fidelity
                }
            # Ajuste adaptativo de ξ
            ξ *= 0.95

        return {'P_AC': 0.0, 'fidelity': fidelity, 'iterations': max_iter}

    def viability_map(self, ξ_range=None, Δt_range=None):
        """
        Gera mapa 2D de P_AC(ξ, Δt)
        """
        if ξ_range is None: ξ_range = np.linspace(0.1, 2.0, 10)
        if Δt_range is None: Δt_range = np.linspace(1e-9, 1e-3, 10)

        results = np.zeros((len(ξ_range), len(Δt_range)))

        for i, ξ in enumerate(ξ_range):
            for j, Δt in enumerate(Δt_range):
                # Estado de teste: coerência |α⟩
                ρ_test = qt.coherent_dm(self.N_dim, 1.0)
                res = self.novikov_consistency(ρ_test, ξ, Δt)
                results[i,j] = res['P_AC']

        return ξ_range, Δt_range, results
