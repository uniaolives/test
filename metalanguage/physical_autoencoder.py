import torch
import torch.nn as nn
from metalanguage.dynamic_expansion import DynamicExpansion

class PhysicalAutoencoder(nn.Module):
    """
    Integrated Model: Sparse Autoencoder + Physical Regressor
    Maps latent dimensions to astrophysical parameters.
    """
    def __init__(self, base_dim=16, max_expansion=64, vocab_size=2, n_phys_params=6):
        super().__init__()
        # Autoencoder component
        self.embedding = nn.Embedding(vocab_size, base_dim)
        self.dynamic_exp = DynamicExpansion(base_dim, max_expansion)
        self.decoder_proj = nn.Linear(max_expansion, base_dim)
        self.output_layer = nn.Linear(base_dim, vocab_size)

        # Physical regression head
        self.phys_regressor = nn.Sequential(
            nn.Linear(max_expansion, 32),
            nn.ReLU(),
            nn.Linear(32, n_phys_params)
        )

    def forward(self, x):
        """
        Forward pass:
        x shape: [batch, seq_len]
        """
        embedded = self.embedding(x)                     # [batch, seq_len, base_dim]
        expanded, factors = self.dynamic_exp(embedded)   # [batch, seq_len, max_exp], factors [batch, seq_len]

        # Reconstruction branch
        decoded = self.decoder_proj(expanded)            # [batch, seq_len, base_dim]
        logits = self.output_layer(decoded)              # [batch, seq_len, vocab_size]

        # Physical regression branch (pooled representation)
        pooled = expanded.mean(dim=1)                     # [batch, max_exp]
        phys_params = self.phys_regressor(pooled)         # [batch, n_phys_params]

        return logits, factors, phys_params, pooled
