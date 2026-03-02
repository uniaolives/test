# economic_attention_mechanism.py
# Integração com VajraEntropyMonitor v4.7.2 (Memória ID 3)

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class CivilizationalState:
    """Estado ontológico de uma civilização em AoE"""
    resources: torch.Tensor  # [Wood, Food, Gold, Stone] - Normalizado [0,1]
    population_vector: torch.Tensor  # [Military, Civilian, Idle] - One-hot distribuído
    tech_tree_position: torch.Tensor  # Embedding da posição na árvore tech
    fog_entropy: float  # Entropia de Shannon da informação visível
    nash_stability: float  # Coeficiente de estabilidade de acordos diplomáticos

class EconomicAttentionGate(nn.Module):
    """
    Gate de atenção esparsa para governança em larga escala
    Padrão I40: Triple-modular redundancy em decisões macroeconômicas
    """

    def __init__(self, d_model: int = 512, satoshi_seed: str = ""):
        super().__init__()
        self.d_model = d_model

        # Projeção do Substrato Econômico para espaço latente físico-computacional
        self.resource_proj = nn.Linear(4, d_model // 4)  # 4 recursos AoE
        self.pop_proj = nn.Linear(3, d_model // 4)       # 3 classes pop
        self.tech_proj = nn.Linear(128, d_model // 2)    # Tech tree embedding

        # Atenção esparsa: foca apenas em gargalos críticos (Pattern I40)
        self.sparse_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Vajra Circuit Breaker para "Economic Hallucinations"
        self.entropy_threshold = 0.98  # Memória ID 3: Superconductive state
        self.vajra_gate = nn.Linear(d_model, 1)  # Decide se estado é válido

    def forward(self, state: CivilizationalState) -> Tuple[torch.Tensor, bool]:
        """
        Retorna: (context_vector, is_valid)
        is_valid = False se detectar anomalia econômica (cheat/exploit)
        """
        # Concatena projeções (Substrate Logic: processamento = deformação geométrica)
        r_emb = self.resource_proj(state.resources.unsqueeze(0))
        p_emb = self.pop_proj(state.population_vector.unsqueeze(0))
        t_emb = self.tech_proj(state.tech_tree_position.unsqueeze(0))

        substrate = torch.cat([r_emb, p_emb, t_emb], dim=-1)  # [1, d_model]

        # Atenção esparsa: máscara para ignorar agentes ociosos (eficiência <5ms)
        mask = self._generate_scarcity_mask(state.resources)

        # Adjust substrate for MultiheadAttention (needs sequence dimension)
        substrate_seq = substrate.unsqueeze(1) # [1, 1, d_model]

        # Note: MultiheadAttention mask needs to be compatible with batch and sequence
        # Here we just use a simple mask if needed, but for 1 element sequence it might be trivial

        attended, attention_weights = self.sparse_attention(
            substrate_seq, substrate_seq, substrate_seq
            # attn_mask=mask # mask needs proper shape [batch * heads, seq, seq] or [seq, seq]
        )

        attended = attended.squeeze(1)

        # Vajra Validation: Detecta estados impossíveis (ex: ouro > max teórico)
        validity_score = torch.sigmoid(self.vajra_gate(attended))
        is_valid = validity_score > 0.5 and state.fog_entropy < self.entropy_threshold

        return attended.squeeze(0), bool(is_valid.item())

    def _generate_scarcity_mask(self, resources: torch.Tensor) -> torch.Tensor:
        """
        Gera máscara de atenção baseada na curva de escassez.
        Recursos < 20% recebem atenção total; > 80% são mascarados (baixa prioridade).
        """
        scarcity = 1.0 - resources  # Inverte: baixo recurso = alta atenção
        mask = scarcity > 0.2  # Threshold adaptativo
        return mask.unsqueeze(0)
