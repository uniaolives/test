# chaos_viz.py
"""
Chaos Trace Visualization (χ_CHAOS_TRACE).
Visualiza o desvio da geodésica urbana em regime de alta entropia.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_chaos_trace(filename='chaos_trace.png'):
    # Criar grade da cidade
    x = np.linspace(0, 10, 100)
    y_safe = np.full_like(x, 5) # Rota prevista (reta)

    # Rota divergente (errática)
    noise = 1.5 * np.sin(x * 2.0) + 0.5 * np.random.randn(len(x))
    y_chaos = y_safe + noise

    plt.figure(figsize=(10, 6), facecolor='#0d0d0d')
    ax = plt.gca()
    ax.set_facecolor('#0d0d0d')

    # Desenhar grid da cidade (sutil)
    for i in range(11):
        plt.axhline(i, color='#1a1a1a', lw=0.5)
        plt.axvline(i, color='#1a1a1a', lw=0.5)

    # Plotar rotas
    plt.plot(x, y_safe, color='#00aaff', lw=2, alpha=0.6, label='Geodésica Prevista (Copacabana)')
    plt.plot(x, y_chaos, color='#ff3333', lw=2.5, label='Traço do Caos (Divergência Lapa)')

    # Ponto atual (Nó de Ouro)
    plt.scatter(x[-1], y_chaos[-1], color='gold', s=100, zorder=5, label='Nó Γ_divergente')

    plt.title('χ_CHAOS_TRACE: A Geodésica do Fogo Roubado', color='white', fontsize=14)
    plt.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    plt.axis('off')

    plt.savefig(filename, dpi=150)
    print(f"Visualização salva em {filename}")

if __name__ == "__main__":
    plot_chaos_trace()
