# src/papercoder_kernel/glp/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import hermite
import math

class HarmonicConfinement(nn.Module):
    """
    Poço harmônico quântico para sequências.
    Estados: |n⟩ com energia E_n = ℏω(n + 1/2)
    No espaço de embedding: polinômios de Hermite × envelope gaussiano
    """
    def __init__(self, max_n=8, sigma=1.0, resolution=256):
        super().__init__()
        self.max_n = max_n  # número de níveis quânticos
        self.sigma = sigma  # largura do poço harmônico
        self.resolution = resolution

        # Autofunções do oscilador harmônico (pré-computadas)
        basis = self._compute_hermite_basis(max_n, resolution)
        self.register_buffer('hermite_basis', basis) # [max_n, resolution]

    def _compute_hermite_basis(self, max_n, resolution):
        x = np.linspace(-3, 3, resolution)
        xi = x / (self.sigma * np.sqrt(2))

        basis = []
        for n in range(max_n):
            H_n = hermite(n)(xi)
            # Normalização: (2^n n! √π)^(-1/2)
            norm = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
            psi = norm * H_n * np.exp(-xi**2 / 2)
            basis.append(psi)

        return torch.tensor(np.stack(basis), dtype=torch.float32)

    def forward(self, positions, amplitudes):
        """
        positions: índices normalizados na sequência [-1, 1], shape [batch, seq_len]
        amplitudes: ocupação de cada modo |n⟩, shape [batch, max_n]
        """
        batch, seq_len = positions.shape
        # Interpolação das autofunções nas posições reais
        # Map [-1, 1] to [0, resolution-1]
        idx = ((positions + 1) / 2 * (self.resolution - 1)).long().clamp(0, self.resolution - 1)

        # basis_sampled: [batch, max_n, seq_len]
        # We use advanced indexing to pick correct basis values for each position in batch
        basis_sampled = self.hermite_basis[:, idx] # [max_n, batch, seq_len]
        basis_sampled = basis_sampled.permute(1, 0, 2) # [batch, max_n, seq_len]

        # Composição coerente dos estados: sum over n (amplitudes[batch, n] * basis_sampled[batch, n, seq_len])
        wavefunction = torch.einsum('bn,bnl->bl', amplitudes, basis_sampled)
        return wavefunction


class SuperlatticeHamiltonian(nn.Module):
    """
    Múltiplos poços harmônicos acoplados.
    Cada escala = modo coletivo do cristal.
    """
    def __init__(self, hidden_dim, scales=[2, 3, 5, 8, 13, 21], coupling_matrix=None):
        """
        Escalas: números de Fibonacci (proporção áurea entre poços)
        """
        super().__init__()
        self.scales = scales
        self.n_wells = len(scales)
        self.hidden_dim = hidden_dim

        # Hamiltoniano de cada poço isolado
        self.wells = nn.ModuleList([
            HarmonicConfinement(max_n=min(s, 8), sigma=s/5.0)
            for s in scales
        ])

        # Learnable occupation amplitudes from input embedding
        self.occupation_nets = nn.ModuleList([
            nn.Linear(hidden_dim, well.max_n)
            for well in self.wells
        ])

        # Matriz de acoplamento (tunelamento entre poços)
        if coupling_matrix is None:
            # Acoplamento decai com diferença de escala
            scales_t = torch.tensor(scales, dtype=torch.float32)
            coupling = torch.exp(-torch.abs(scales_t.unsqueeze(0) - scales_t.unsqueeze(1)) / 2.0)
            coupling = coupling - torch.diag(torch.diag(coupling))
        else:
            coupling = coupling_matrix

        self.register_buffer('coupling', coupling)

        # Frequências próprias de cada poço (análogo a ℏω_i)
        self.omega = nn.Parameter(
            torch.tensor([1.0/s for s in scales])
        )

    def forward(self, sequence_embedding):
        """
        sequence_embedding: [batch, seq_len, dim]
        """
        batch, seq_len, dim = sequence_embedding.shape
        device = sequence_embedding.device

        # Posições normalizadas no poço harmônico
        positions = torch.linspace(-1, 1, seq_len, device=device).unsqueeze(0).expand(batch, -1)

        # Ocupação de cada modo em cada poço
        # Pooling global da sequência para determinar amplitudes
        pooled = sequence_embedding.mean(dim=1) # [batch, dim]

        well_states = []
        for i, well in enumerate(self.wells):
            amp = self.occupation_nets[i](pooled)
            amp_softmax = F.softmax(amp, dim=-1)
            # well(positions, amp) -> [batch, seq_len]
            # We want to return states with hidden_dim
            wave = well(positions, amp_softmax) # [batch, seq_len]
            # Expande para hidden_dim (cada poço contribui com seu estado para o campo latente)
            # Simulação: cada escala modula o campo de informação
            well_states.append(wave.unsqueeze(-1) * pooled.unsqueeze(1)) # [batch, seq_len, dim]

        return torch.stack(well_states, dim=1)  # [batch, n_wells, seq_len, dim]


class ResonantTunnelingAttention(nn.Module):
    """
    Tunelamento ressonante como mecanismo de atenção.
    """
    def __init__(self, n_wells, hidden_dim, temperature=0.1):
        super().__init__()
        self.n_wells = n_wells
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Matriz S de espalhamento
        self.S_matrix = nn.Parameter(
            torch.randn(n_wells, n_wells, hidden_dim) * 0.1
        )

        # Fase de Breit-Wigner para ressonâncias
        self.resonance_energy = nn.Parameter(torch.randn(n_wells, hidden_dim))
        self.resonance_width = nn.Parameter(torch.ones(n_wells, hidden_dim) * 0.1)

    def breit_wigner(self, E, E_0, Gamma):
        """Amplitude de transmissão perto de ressonância."""
        # Using a real approximation for the prototype
        # Re(BW) = Gamma^2 / ((E-E0)^2 + (Gamma/2)^2)
        return Gamma / ((E - E_0)**2 + (Gamma/2.0)**2 + 1e-8)

    def forward(self, well_states, query_energy=None):
        """
        well_states: [batch, n_wells, seq_len, hidden_dim]
        """
        batch, n_wells, seq_len, hidden = well_states.shape

        if query_energy is None:
            query_energy = well_states.mean(dim=[1, 2])  # [batch, hidden]

        # Expandir para formato de ressonância
        E = query_energy.unsqueeze(1)  # [batch, 1, hidden]
        E_0 = self.resonance_energy.unsqueeze(0)  # [1, n_wells, hidden]
        Gamma = torch.abs(self.resonance_width).unsqueeze(0)

        # Amplitude de tunelamento (Breit-Wigner)
        tunneling_amp = self.breit_wigner(E, E_0, Gamma)  # [batch, n_wells, hidden]
        tunneling_amp = F.normalize(tunneling_amp, p=1, dim=1)

        # Matriz S para mistura entre poços
        S = F.softmax(self.S_matrix / self.temperature, dim=1) # [n_wells, n_wells, hidden]

        # Estado tunelado: superposição coerente
        # mixed_states[batch, i, seq, h] = sum_j S[i, j, h] * well_states[batch, j, seq, h]
        mixed_states = torch.einsum('ijh,bjsh->bish', S, well_states)

        # Ponderação pela amplitude de tunelamento
        output = mixed_states * tunneling_amp.unsqueeze(2)

        return output, tunneling_amp


class BCD_GLPLinearA(nn.Module):
    """
    GLP completo: B*C*D = Harmônico × Superlattice × Tunelamento
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, scales=[2, 3, 5, 8, 13, 21]):
        super().__init__()
        self.scales = scales
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)

        # B*C: Superlattice de múltiplas escalas harmônicas
        self.hamiltonian = SuperlatticeHamiltonian(
            hidden_dim=hidden_dim,
            scales=scales
        )

        # D: Tunelamento ressonante
        self.tunneling = ResonantTunnelingAttention(
            n_wells=len(scales),
            hidden_dim=hidden_dim
        )

        # Observáveis
        self.sign_predictor = nn.Linear(hidden_dim, vocab_size)
        self.geometry_probe = nn.Linear(hidden_dim, 3)

    def forward(self, sign_ids, return_wavefunction=False):
        # Embedding
        x = self.embedding(sign_ids) # [batch, seq_len, embed_dim]
        x = self.proj(x) # [batch, seq_len, hidden_dim]

        # Superlattice (B*C)
        well_states = self.hamiltonian(x) # [batch, n_wells, seq_len, hidden_dim]

        # Tunelamento (D)
        tunneled, probs = self.tunneling(well_states)

        # Colapso (Soma coerente)
        final_state = tunneled.sum(dim=1) # [batch, seq_len, hidden_dim]
        tablet_repr = final_state.mean(dim=1) # [batch, hidden_dim]

        output = {
            'tablet_repr': tablet_repr,
            'integrated_state': final_state,
            'sign_logits': self.sign_predictor(final_state),
            'geometry': self.geometry_probe(tablet_repr),
            'scale_probabilities': probs,
            'tunneling_strength': probs.std(dim=1).mean()
        }

        if return_wavefunction:
            output['well_states'] = well_states
            output['tunneled_states'] = tunneled

        return output
