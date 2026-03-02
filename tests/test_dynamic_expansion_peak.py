import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.dynamic_expansion import DynamicExpansion

def local_entropy(seq, window=11):
    entropias = []
    for i in range(len(seq)):
        # define janela centrada em i (com limites)
        left = max(0, i - window//2)
        right = min(len(seq), i + window//2 + 1)
        window_seq = seq[left:right]
        # calcula distribuição de 0s e 1s na janela
        p0 = window_seq.count(0) / len(window_seq)
        p1 = window_seq.count(1) / len(window_seq)
        # entropia de Shannon (bits)
        e = entropy([p0, p1], base=2)
        entropias.append(e)
    return entropias

def run_peak_experiment():
    # 1. Sequência de 85 bits (original)
    bits_original = "00001010111011000111110011010010000101011101100011111001101001000010101110"
    bits_original = [int(b) for b in bits_original]   # lista de inteiros
    seq_len = len(bits_original)

    # 2. Criar uma versão com pico de entropia artificial
    #    Aumentar a aleatoriedade nos bits 40 a 50
    bits_peak = bits_original.copy()
    np.random.seed(42)
    for i in range(40, 51):                     # janela de 11 bits
        if i < len(bits_peak):
            # Troca o bit por um valor aleatório (0 ou 1)
            bits_peak[i] = np.random.randint(0, 2)

    entropy_orig = local_entropy(bits_original, window=11)
    entropy_peak = local_entropy(bits_peak, window=11)

    # 3. Preparar os dados para o modelo
    base_dim = 16
    embedding = nn.Embedding(2, base_dim)   # 2 símbolos: 0 e 1

    # Converte listas para tensores
    tokens_orig = torch.tensor(bits_original).unsqueeze(0)   # [1, seq_len]
    tokens_peak = torch.tensor(bits_peak).unsqueeze(0)

    # Obtém embeddings
    x_orig = embedding(tokens_orig)          # [1, seq_len, base_dim]
    x_peak = embedding(tokens_peak)          # [1, seq_len, base_dim]

    # 4. Instanciar a camada DynamicExpansion e aplicar
    max_exp = 64
    # Fix seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    embedding = nn.Embedding(2, base_dim)   # 2 símbolos: 0 e 1
    dynamic_layer = DynamicExpansion(base_dim, max_expansion=max_exp)

    # Initialize weights to something non-zero so we see changes
    nn.init.normal_(dynamic_layer.entropy_estimator.weight)
    nn.init.normal_(dynamic_layer.entropy_estimator.bias)

    with torch.no_grad():
        x_exp_orig, factors_orig = dynamic_layer(x_orig)
        x_exp_peak, factors_peak = dynamic_layer(x_peak)

    factors_orig = factors_orig.squeeze(0).cpu().numpy()
    factors_peak = factors_peak.squeeze(0).cpu().numpy()

    # 5. Visualização (Salvar em arquivo)
    plt.figure(figsize=(14, 10))

    # Subplot 1: entropia local
    plt.subplot(3,1,1)
    plt.plot(entropy_orig, label='Original', color='blue', alpha=0.7)
    plt.plot(entropy_peak, label='Com pico artificial', color='red', alpha=0.7)
    plt.axvspan(40, 50, color='gray', alpha=0.3, label='Região do pico')
    plt.xlabel('Posição do bit')
    plt.ylabel('Entropia local (bits)')
    plt.title('Entropia local (janela 11)')
    plt.legend()
    plt.grid(True)

    # Subplot 2: fator de expansão para a sequência original
    plt.subplot(3,1,2)
    plt.stem(range(seq_len), factors_orig, basefmt=" ", linefmt='blue', markerfmt='bo', label='Original')
    plt.axvspan(40, 50, color='gray', alpha=0.3)
    plt.xlabel('Posição do bit')
    plt.ylabel('Fator de expansão')
    plt.title('Expansão dimensional - Sequência original')
    plt.ylim(0, max_exp+5)
    plt.grid(True)

    # Subplot 3: fator de expansão para a sequência com pico
    plt.subplot(3,1,3)
    plt.stem(range(seq_len), factors_peak, basefmt=" ", linefmt='red', markerfmt='ro', label='Com pico')
    plt.axvspan(40, 50, color='gray', alpha=0.3)
    plt.xlabel('Posição do bit')
    plt.ylabel('Fator de expansão')
    plt.title('Expansão dimensional - Sequência com pico de entropia')
    plt.ylim(0, max_exp+5)
    plt.grid(True)

    plt.tight_layout()
    plot_path = 'entropy_peak_analysis.png'
    plt.savefig(plot_path)
    print(f"✅ Visualization saved to {plot_path}")

    # 6. Análise comparativa
    print("\n=== Estatísticas dos fatores de expansão ===")
    print(f"Original: média = {factors_orig.mean():.2f}, desvio = {factors_orig.std():.2f}")
    print(f"Com pico: média = {factors_peak.mean():.2f}, desvio = {factors_peak.std():.2f}")

    regiao_orig = factors_orig[40:51]
    regiao_peak = factors_peak[40:51]
    print(f"\nNa região do pico (bits 40-50):")
    print(f"  Original: média = {regiao_orig.mean():.2f}, desvio = {regiao_orig.std():.2f}")
    print(f"  Com pico: média = {regiao_peak.mean():.2f}, desvio = {regiao_peak.std():.2f}")

if __name__ == "__main__":
    run_peak_experiment()
