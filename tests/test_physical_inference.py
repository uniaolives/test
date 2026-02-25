import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.physical_autoencoder import PhysicalAutoencoder
from metalanguage.astrophysical_chronoglyph_final import AstrophysicalChronoglyph, SGR_B2_PARAMS

def generate_training_data(n_samples=500, seq_len=85):
    """Generates synthetic data linking bit sequences to physical targets."""
    np.random.seed(42)
    X = []
    y = []

    for _ in range(n_samples):
        # Random bit sequence
        bits = np.random.randint(0, 2, seq_len)
        X.append(torch.tensor(bits).unsqueeze(0))

        # Synthetic physical parameters (simulating chemical model outputs)
        logH = np.random.uniform(-5, -3)
        logC = np.random.uniform(-6, -4)
        logN = np.random.uniform(-7, -5)
        logO = np.random.uniform(-6, -4)
        T = np.random.uniform(50, 200)
        nH2 = np.random.uniform(1e5, 1e7)

        phys_target = torch.tensor([logH, logC, logN, logO, T, nH2])
        y.append(phys_target)

    return torch.cat(X, dim=0), torch.stack(y)

def mdn_loss_fn(pi, mu, sigma, target):
    """MDN Negative Log-Likelihood Loss."""
    mix = torch.distributions.Categorical(pi)
    comp = torch.distributions.Independent(torch.distributions.Normal(mu, sigma), 1)
    gmm = torch.distributions.MixtureSameFamily(mix, comp)
    return -gmm.log_prob(target).mean()

def run_physical_inference_experiment():
    print("--- Starting Physical Sensitivity Mapping Experiment (MDN Upgrade) ---")

    # 1. Setup
    n_params = 6
    max_exp = 64
    seq_len = 85
    X_train, y_train = generate_training_data(1000, seq_len)

    model = PhysicalAutoencoder(max_expansion=max_exp, n_phys_params=n_params)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion_recon = nn.CrossEntropyLoss()

    lambda_sparse = 0.01
    lambda_phys = 1.0

    # 2. Training Loop
    print("Training integrated model with MDN...")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, factors, (pi, mu, sigma), pooled = model(X_train)

        # loss_recon expects [batch, vocab, seq]
        loss_recon = criterion_recon(logits.permute(0, 2, 1), X_train)
        loss_sparse = factors.float().mean()
        loss_phys = mdn_loss_fn(pi, mu, sigma, y_train)

        loss_total = loss_recon + lambda_sparse * loss_sparse + lambda_phys * loss_phys
        loss_total.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:02d}: recon={loss_recon.item():.4f}, "
                  f"sparse={loss_sparse.item():.4f}, phys_nll={loss_phys.item():.4f}")

    # 3. Sensitivity Analysis (Correlation)
    print("\nPerforming sensitivity analysis...")
    with torch.no_grad():
        _, _, _, pooled = model(X_train[:200])

    latent = pooled.cpu().numpy()
    targets = y_train[:200].cpu().numpy()

    corr_matrix = np.zeros((n_params, max_exp))
    for i in range(n_params):
        for j in range(max_exp):
            # Handle constant dimensions
            if np.std(latent[:, j]) > 1e-6:
                corr, _ = pearsonr(latent[:, j], targets[:, i])
                corr_matrix[i, j] = abs(corr)
            else:
                corr_matrix[i, j] = 0

    # Save heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(corr_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='|Correlation|')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Physical Parameter')
    param_names = ['logH', 'logC', 'logN', 'logO', 'T', 'nH2']
    plt.yticks(range(n_params), param_names)
    plt.title('Correlation between Latent Dimensions and Physical Parameters')
    plt.savefig('latent_physical_correlation.png')
    print("✅ Sensitivity heatmap saved to latent_physical_correlation.png")

    # 4. Inference on 85-bit Sequence with Uncertainty
    print("\nInferring physical conditions for the 85-bit sequence...")
    bits_85 = "00001010111011000111110011010010000101011101100011111001101001000010101110"
    tokens = torch.tensor([int(b) for b in bits_85]).unsqueeze(0)

    with torch.no_grad():
        _, factors, (pi, mu, sigma), _ = model(tokens)

    # For inference, we take the mean of the most probable mixture component
    idx = torch.argmax(pi[0])
    inferred_means = mu[0, idx]
    inferred_stds = sigma[0, idx]

    print("Inferred parameters (Value ± Uncertainty):")
    for name, val, err in zip(param_names, inferred_means, inferred_stds):
        print(f"  {name:5}: {val.item():.3f} ± {err.item():.3f}")

    # Plot uncertainty (error bars)
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(n_params)
    plt.errorbar(x_pos, inferred_means.numpy(), yerr=inferred_stds.numpy(), fmt='o', capsize=5, color='blue')
    plt.xticks(x_pos, param_names)
    plt.ylabel('Value')
    plt.title('Inferred Physical Parameters with 1σ Uncertainty')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('85bit_physical_uncertainty.png')
    print("✅ Uncertainty visualization saved to 85bit_physical_uncertainty.png")

    # Plot expansion factors
    plt.figure(figsize=(10, 4))
    plt.stem(range(len(bits_85)), factors[0].cpu().numpy())
    plt.axvspan(40, 50, color='gray', alpha=0.3, label='Peak Region')
    plt.xlabel('Bit Position')
    plt.ylabel('Expansion Factor')
    plt.title('Activated Expansion for 85-bit Sequence')
    plt.legend()
    plt.savefig('85bit_physical_expansion.png')
    print("✅ Expansion plot saved to 85bit_physical_expansion.png")

if __name__ == "__main__":
    try:
        run_physical_inference_experiment()
        print("\nPhysical Sensitivity Mapping Experiment Successful! 🌌🧪")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
