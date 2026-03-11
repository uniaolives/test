# orbvm_pinn.py

import torch
import torch.nn as nn

class OrbVM_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Rede neural profunda
        self.net = nn.Sequential(
            nn.Linear(4, 128),  # x, y, z, t
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)   # Ψ (Potencial Arkhe)
        )

    def forward(self, x, y, z, t):
        # Ensure inputs have requires_grad=True if they don't already
        return self.net(torch.cat([x, y, z, t], dim=1))

    def compute_information_density(self, x, y, z, t):
        """
        Calcula a densidade de informação ρ_info baseada no estado do campo.
        """
        # Placeholder para a densidade de informação do Arkhe
        return torch.sin(x) * torch.cos(t) # Exemplo de flutuação de vácuo

    def physics_loss(self, x, y, z, t):
        """
        O coração da consciência.
        A rede é forçada a obedecer a equação de Whittaker.
        """
        # Ensure inputs require grad for autograd
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)

        psi = self.forward(x, y, z, t)

        # Derivadas parciais (autograd)
        psi_t = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_tt = torch.autograd.grad(psi_t, t, grad_outputs=torch.ones_like(psi_t), create_graph=True)[0]

        # Laplaciano espacial
        psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]

        psi_y = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_yy = torch.autograd.grad(psi_y, y, grad_outputs=torch.ones_like(psi_y), create_graph=True)[0]

        psi_z = torch.autograd.grad(psi, z, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_zz = torch.autograd.grad(psi_z, z, grad_outputs=torch.ones_like(psi_z), create_graph=True)[0]

        laplacian = psi_xx + psi_yy + psi_zz

        # Equação de Whittaker: □Ψ + μ²Ψ = ρ
        mu_sq = 1e-34  # Constante de Arkhe
        rho_info = self.compute_information_density(x, y, z, t)

        residual = psi_tt - laplacian + mu_sq * psi - rho_info

        return torch.mean(residual ** 2)
