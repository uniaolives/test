"""
neural_world_model.py
Modelo de mundo neural autoregressivo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
import math

class NeuralWorldModel(nn.Module):
    """Modelo de mundo neural com atenção espaço-temporal"""

    def __init__(self,
                 obs_dim=256,
                 action_dim=32,
                 hidden_dim=1024,
                 num_layers=12):
        super().__init__()

        # Encoder multimodal
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Transformer temporal
        config = GPT2Config(
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=16,
            n_positions=1024,
            n_ctx=1024,
        )
        self.transformer = GPT2Model(config)

        # Decoders para diferentes modalidades
        self.obs_decoder = nn.Linear(hidden_dim, obs_dim)
        self.reward_decoder = nn.Linear(hidden_dim, 1)
        self.done_decoder = nn.Linear(hidden_dim, 1)

        # Modelos de física implícita
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),  # Parâmetros físicos
        )

    def forward(self, observations, actions, timesteps):
        """Prediz próximo estado do mundo"""

        # Codifica observações e ações
        obs_emb = self.obs_encoder(observations)
        act_emb = self.action_encoder(actions)

        # Combina com embeddings temporais
        time_emb = self._timestep_embedding(timesteps, dim=obs_emb.shape[-1])

        # Input para transformer
        inputs = obs_emb + act_emb + time_emb

        # Processa com transformer
        transformer_out = self.transformer(
            inputs_embeds=inputs.unsqueeze(0)
        ).last_hidden_state

        # Decodifica predições
        next_obs_pred = self.obs_decoder(transformer_out)
        reward_pred = self.reward_decoder(transformer_out)
        done_pred = torch.sigmoid(self.done_decoder(transformer_out))

        # Extrai física implícita
        physics_params = self.physics_head(transformer_out)

        return {
            'next_obs': next_obs_pred,
            'reward': reward_pred,
            'done': done_pred,
            'physics': physics_params,
        }

    def predict_rollout(self, initial_obs, action_sequence, steps=50):
        """Prediz sequência de estados futuros"""

        current_obs = initial_obs
        rollout = []

        for t in range(steps):
            # Prediz próximo estado
            with torch.no_grad():
                pred = self.forward(
                    current_obs,
                    action_sequence[t],
                    torch.tensor([t]).to(current_obs.device)
                )

            # Armazena predição
            rollout.append({
                'obs': pred['next_obs'],
                'reward': pred['reward'],
                'physics': pred['physics'],
            })

            # Atualiza estado atual
            current_obs = pred['next_obs']

        return rollout

    def _timestep_embedding(self, timesteps, dim=1024, max_period=10000):
        """Embedding sinusoidal para timesteps"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding
