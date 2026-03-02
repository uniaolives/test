import torch
import torch.nn as nn

class FrechetTensor(nn.Module):
    """
    Representa um tensor em um espaço de Fréchet, definido por uma família de seminormas.
    O tensor se ajusta para satisfazer múltiplas escalas de medida simultaneamente.
    """
    def __init__(self, data, seminorms):
        super().__init__()
        # O dado vive em um estado latente sem norma fixa
        self.latent_space = nn.Parameter(data)
        # Uma lista de funções que definem "réguas" de medida (seminorms)
        self.seminorms = seminorms

    def get_convergence_vector(self):
        """
        Retorna o perfil de convergência para cada escala de medida
        da família de Fréchet.
        """
        return torch.stack([rho(self.latent_space) for rho in self.seminorms])

    def forward(self, x):
        """
        A operação é uma tradução de escala.
        """
        return self.latent_space * x

def seminorm_k(t, k):
    """
    Exemplo de seminorma: a derivada de ordem k como medida de 'tamanho' ou suavidade.
    """
    if k == 0:
        return torch.norm(t)

    # Diferença finita como aproximação de derivada
    diff_t = t
    for _ in range(k):
        if diff_t.size(0) <= 1:
            return torch.tensor(0.0)
        diff_t = torch.diff(diff_t)

    return torch.norm(diff_t)
