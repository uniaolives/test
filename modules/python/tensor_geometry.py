"""
Módulo de Geometria Tensorial Avançada
Foco: Variedades Hiperbólicas (Poincaré) e Operações de Möbius
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoincareBall:
    """
    Implementação da bola de Poincaré para tensores.
    Permite operações em hiperespaços curvos (ℍ³).
    """
    def __init__(self, c: float = 1.0):
        self.c = c  # Curvatura

    def project(self, x: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
        """Projeta pontos de volta para dentro da bola unitária"""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = (1 - epsilon) / (self.c**0.5)
        cond = norm > max_norm
        projected = x / (norm + 1e-10) * max_norm
        return torch.where(cond, projected, x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Soma de Möbius vetorizada: x ⊕ y"""
        x2 = torch.sum(x**2, dim=-1, keepdim=True)
        y2 = torch.sum(y**2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2

        return num / (denom + 1e-10)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Mapeamento exponencial: ponto x, vetor tangente v"""
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        lambda_x = 2 / (1 - self.c * torch.sum(x**2, dim=-1, keepdim=True) + 1e-10)

        res = torch.tanh(self.c**0.5 * lambda_x * v_norm / 2) * v / (self.c**0.5 * v_norm + 1e-10)
        return self.mobius_add(x, res)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mapeamento logarítmico de x para y"""
        add_neg_x = self.mobius_add(-x, y)
        norm_add = torch.norm(add_neg_x, p=2, dim=-1, keepdim=True)
        lambda_x = 2 / (1 - self.c * torch.sum(x**2, dim=-1, keepdim=True) + 1e-10)

        return 2 / (self.c**0.5 * lambda_x + 1e-10) * torch.atanh(self.c**0.5 * norm_add) * add_neg_x / (norm_add + 1e-10)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distância geodésica na bola de Poincaré"""
        sq_dist = torch.sum((x - y)**2, dim=-1)
        x_sq = torch.sum(x**2, dim=-1)
        y_sq = torch.sum(y**2, dim=-1)

        arg = 1 + 2 * self.c * sq_dist / ((1 - self.c * x_sq) * (1 - self.c * y_sq) + 1e-10)
        return torch.acosh(arg) / (self.c**0.5)

class HyperbolicLayer(nn.Module):
    """Camada Linear Hyperbólica (Möbius Linear)"""
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.ball = PoincareBall(c)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Garante que x está na bola antes de log_map
        x = self.ball.project(x)
        # Mapeia para o espaço tangente, aplica linear, mapeia de volta
        x_tangent = self.ball.log_map(torch.zeros_like(x), x)
        y_tangent = F.linear(x_tangent, self.weight, self.bias)
        return self.ball.exp_map(torch.zeros_like(y_tangent), y_tangent)
