# src/papercoder_kernel/glp/training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumActionLoss(nn.Module):
    """
    Perda baseada no funcional de ação.
    Minimiza energia cinética + potencial + termo de tunelamento.
    """
    def __init__(self, alpha_kinetic=1.0, alpha_potential=1.0, alpha_tunnel=0.5):
        super().__init__()
        self.alpha_kinetic = alpha_kinetic
        self.alpha_potential = alpha_potential
        self.alpha_tunnel = alpha_tunnel

    def forward(self, outputs, targets, model_states=None):
        """
        outputs: dict do modelo BCD_GLPLinearA
        targets: [batch, seq_len] IDs dos signos
        model_states: dict com wavefunction se disponível
        """
        # Perda de predição (energia potencial do erro)
        logits = outputs['sign_logits']
        potential = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0 # <PAD>
        )

        kinetic = torch.tensor(0.0, device=logits.device)
        if model_states is not None and 'tunneled_states' in model_states:
            # Regularização cinética: suavidade das transições
            states = model_states['tunneled_states']
            # states: [batch, n_wells, seq_len, hidden]
            kinetic = ((states[:, :, 1:, :] - states[:, :, :-1, :])**2).mean()

        # Termo de tunelamento: incentiva coerência entre escalas
        tunnel_energy = -torch.log(outputs['tunneling_strength'] + 1e-8)

        total = (self.alpha_potential * potential +
                self.alpha_kinetic * kinetic +
                self.alpha_tunnel * tunnel_energy)

        return total, {
            'potential': potential.item(),
            'kinetic': kinetic.item(),
            'tunnel': tunnel_energy.item()
        }

def analyze_confinement(cooc_matrix):
    """
    Diagonaliza M* e verifica se espectro é consistente com quantum dot.
    """
    eigenvals = np.linalg.eigvalsh(cooc_matrix)
    # Filtra autovalores muito pequenos (ruído)
    eigenvals = eigenvals[eigenvals > 1e-5]

    if len(eigenvals) < 3:
        return {'mean_spacing_ratio': 0.0, 'confinement_regime': 'unknown'}

    # Spacing ratio: r_n = (E_{n+1} - E_n) / (E_n - E_{n-1})
    spacings = np.diff(eigenvals)
    # Evita divisão por zero
    ratios = spacings[1:] / (spacings[:-1] + 1e-8)

    mean_ratio = np.mean(ratios)

    # Regime detection
    # harmonic: r ~ 1 (níveis equiespaçados)
    # square well: r cresce (n^2)
    regime = 'harmonic' if 0.7 < mean_ratio < 1.3 else 'non-harmonic'

    return {
        'mean_spacing_ratio': float(mean_ratio),
        'confinement_regime': regime,
        'spectral_gap': float(eigenvals[-1] - eigenvals[-2]) if len(eigenvals) >= 2 else 0.0
    }

def train_step(model, optimizer, sign_ids, loss_fn):
    model.train()
    optimizer.zero_grad()

    outputs = model(sign_ids, return_wavefunction=True)
    # targets are shifted sign_ids for next sign prediction, or same for reconstruction
    # In this simple case, we use sign_ids as targets for the classifier head
    loss, loss_dict = loss_fn(outputs, sign_ids, outputs)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_dict
