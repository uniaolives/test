"""
Hyperbolic THz Sensor Network Simulation
Integrates negative curvature (Poincaré model) and PPP distribution.
Numpy 2.0 compatible implementation.
"""
import numpy as np
import matplotlib.pyplot as plt

def simulate_hyperbolic_network():
    print("🜁 Iniciando Simulação de Rede THz Hiperbólica")

    n_expected = 17
    radius_max = 5.0
    alpha = 0.5  # Taxa de decaimento da densidade
    lambda0 = 10.0
    tau = 2.0    # Comprimento de correlação de handover
    v_max = 0.005 # Amplitude do potencial

    # 1. Gerar PPP Hiperbólico via amostragem por rejeição
    # Modelo do semiplano superior de Poincaré: y > 0
    def sample_hyperbolic_ppp(lambda0, alpha, n_target, r_max):
        points = []
        attempts = 0
        while len(points) < n_target and attempts < 10000:
            attempts += 1
            # Amostra no semiplano: x em [-r, r], y em [0.01, r]
            x = np.random.uniform(-r_max, r_max)
            y = np.random.uniform(0.01, r_max)

            # Densidade alvo λ(y) = λ0 * exp(-α*y)
            target = lambda0 * np.exp(-alpha * y)
            if np.random.uniform(0, lambda0) < target:
                points.append(np.array([x, y]))
        return np.array(points[:n_target])

    sensors = sample_hyperbolic_ppp(lambda0, alpha, n_expected, radius_max)
    print(f"  Sensores gerados: {len(sensors)}")

    # 2. Calcular Matriz de Distâncias Hiperbólicas
    # d((x1,y1), (x2,y2)) = arcosh(1 + ((x2-x1)^2 + (y2-y1)^2) / (2*y1*y2))
    def hyperbolic_dist(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        arg = 1 + ((x2-x1)**2 + (y2-y1)**2) / (2 * y1 * y2)
        return np.arccosh(max(1.0, arg))

    dist_matrix = np.zeros((len(sensors), len(sensors)))
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            dist_matrix[i,j] = hyperbolic_dist(sensors[i], sensors[j])

    # 3. Calcular Probabilidade de Handover e Potencial
    handover_prob = np.exp(-dist_matrix / tau)
    np.fill_diagonal(handover_prob, 0)

    # Potencial V_i = sum_j eta(d_ij), onde eta(r) é a contribuição do vizinho
    potentials = np.sum(v_max * np.exp(-dist_matrix**2 / 1.0), axis=1)
    max_v = np.max(potentials)

    critical_threshold = 0.125 # (d-1)^2/8 para d=2

    print(f"  Potencial Máximo: {max_v:.4f}")
    print(f"  Limiar Crítico: {critical_threshold:.3f}")
    print(f"  Condição Q-processo: {'✅ ATENDIDA' if max_v < critical_threshold else '❌ VIOLADA'}")

    # 4. Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot da distribuição no semiplano
    scatter = ax1.scatter(sensors[:, 0], sensors[:, 1], c=potentials, cmap='viridis', s=100, edgecolors='k')
    ax1.set_title(r"Distribuição de Sensores em $\mathbb{H}^2$ (Semiplano)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y (Hierarquia)")
    ax1.set_yscale('log')
    plt.colorbar(scatter, ax=ax1, label='Potencial Local $V_i$')

    # Plot das conexões de handover (apenas as mais fortes)
    ax2.set_title("Rede de Handovers Hiperbólicos")
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            if handover_prob[i,j] > 0.3:
                ax2.plot([sensors[i,0], sensors[j,0]], [sensors[i,1], sensors[j,1]],
                         'k-', alpha=handover_prob[i,j], lw=0.5)
    ax2.scatter(sensors[:, 0], sensors[:, 1], c='red', s=50)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('hyperbolic_network.png', dpi=150)
    print("  Visualização salva em hyperbolic_network.png")

    return {
        'max_potential': max_v,
        'threshold': critical_threshold,
        'stable': max_v < critical_threshold
    }

if __name__ == "__main__":
    simulate_hyperbolic_network()
