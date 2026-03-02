import torch
import torch.nn as nn
from metalanguage.dynamic_expansion import DynamicExpansion

class MDNHead(nn.Module):
    """
    Mixture Density Network Head for physical parameter regression.
    Outputs parameters for a Gaussian Mixture Model (GMM).
    """
    def __init__(self, input_dim, n_phys_params, n_mixtures=5):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.n_params = n_phys_params
        # Output: mixing coefficients (pi), means (mu), and log-standard deviations (log_sigma)
        # Size: K (weights) + K*D (means) + K*D (stds)
        self.net = nn.Linear(input_dim, n_mixtures * (1 + n_phys_params * 2))

    def forward(self, x):
        batch_size = x.size(0)
        out = self.net(x)

        # Split output into pi, mu, and sigma
        pi_logits = out[:, :self.n_mixtures]
        mu = out[:, self.n_mixtures : self.n_mixtures * (1 + self.n_params)]
        log_sigma = out[:, self.n_mixtures * (1 + self.n_params) :]

        pi = torch.softmax(pi_logits, dim=1)
        mu = mu.view(batch_size, self.n_mixtures, self.n_params)
        # Using softplus for sigma to ensure positivity and numerical stability
        sigma = torch.nn.functional.softplus(log_sigma).view(batch_size, self.n_mixtures, self.n_params) + 1e-4

        return pi, mu, sigma

class PhysicalAutoencoder(nn.Module):
    """
    Integrated Model: Sparse Autoencoder + MDN Head
    Maps latent dimensions to astrophysical parameter distributions.
    """
    def __init__(self, base_dim=16, max_expansion=64, vocab_size=2, n_phys_params=6, n_mixtures=5):
        super().__init__()
        # Autoencoder component
        self.embedding = nn.Embedding(vocab_size, base_dim)
        self.dynamic_exp = DynamicExpansion(base_dim, max_expansion)
        self.decoder_proj = nn.Linear(max_expansion, base_dim)
        self.output_layer = nn.Linear(base_dim, vocab_size)

        # Physical MDN Head (replaces point regressor)
        self.mdn_head = MDNHead(max_expansion, n_phys_params, n_mixtures)

    def forward(self, x):
        """
        Forward pass:
        x shape: [batch, seq_len]
        Returns: logits, expansion factors, MDN parameters (pi, mu, sigma), and pooled latent vector
        """
        embedded = self.embedding(x)                     # [batch, seq_len, base_dim]
        expanded, factors = self.dynamic_exp(embedded)   # [batch, seq_len, max_exp], factors [batch, seq_len]

        # Reconstruction branch
        decoded = self.decoder_proj(expanded)            # [batch, seq_len, base_dim]
        logits = self.output_layer(decoded)              # [batch, seq_len, vocab_size]

        # Physical branch: pooling sequence dimension
        pooled = expanded.mean(dim=1)                     # [batch, max_exp]
        pi, mu, sigma = self.mdn_head(pooled)

        return logits, factors, (pi, mu, sigma), pooled
