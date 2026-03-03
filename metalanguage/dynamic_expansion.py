import torch
import torch.nn as nn

class DynamicExpansion(nn.Module):
    """
    Camada DynamicExpansion: Adapta a dimensionalidade latente à complexidade informacional.
    Usa um estimador de entropia para determinar o fator de expansão de cada token.
    """
    def __init__(self, base_dim, max_expansion=256):
        super().__init__()
        self.base_dim = base_dim
        self.max_expansion = max_expansion
        # Rede que estima a entropia (ou complexidade) a partir do embedding
        self.entropy_estimator = nn.Linear(base_dim, 1)
        # Projeção linear para o espaço expandido (máximo)
        self.expand_proj = nn.Linear(base_dim, max_expansion, bias=False)

    def forward(self, x):
        """
        Forward pass:
        x shape: [batch, seq_len, base_dim]
        Retorna: (x_expanded, expansion_factors)
        """
        # Estima entropia por token
        entropia_logit = self.entropy_estimator(x)          # [batch, seq_len, 1]
        entropia = torch.sigmoid(entropia_logit)            # entre 0 e 1

        # Fator de expansão contínuo (arredondado para inteiro)
        expansion_factor = (1 + entropia * (self.max_expansion - 1)).int()
        # expansion_factor shape: [batch, seq_len, 1]

        # Projeta todos os tokens para o espaço máximo
        x_expanded = self.expand_proj(x)                    # [batch, seq_len, max_expansion]

        # Cria máscara para zeroar dimensões não utilizadas
        batch, seq_len, _ = x_expanded.shape
        # Cria tensor de índices [0,1,...,max_expansion-1]
        indices = torch.arange(self.max_expansion, device=x.device).view(1,1,-1)
        mask = (indices < expansion_factor).float()          # [batch, seq_len, max_expansion]
        x_expanded = x_expanded * mask

        return x_expanded, expansion_factor.squeeze(-1)      # retorna fatores por token
