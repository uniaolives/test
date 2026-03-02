# gossip_coherence_sim.py
# Simulação ANL (pseudo-código representando a execução)
import numpy as np

# Embeddings dos âncoras (r, theta, z)
embeddings = {
    "us-east4-a": (0.5, 1.04, 1.0),
    "us-east4-b": (0.5, 1.05, 1.0),
    "europe-west2-a": (0.8, 0.00, 1.0),
    "europe-west2-b": (0.8, 0.01, 1.0),
    "asia-southeast1-a": (0.9, 1.80, 1.0),
    "asia-southeast1-b": (0.9, 1.81, 1.0),
}

# Função de distância hiperbólica
def d_h(e1, e2):
    r1, th1, z1 = e1
    r2, th2, z2 = e2
    dr = r1 - r2
    dth = (th1 - th2) % (2*np.pi)
    dz = z1 - z2
    numerator = dr*dr + r1*r2*(1 - np.cos(dth)) + dz*dz
    denominator = 2*z1*z2

    # Stability check for arccosh
    val = 1 + numerator/denominator
    return np.arccosh(max(1.0, val))

def run_sim():
    print("Running Hyperbolic Gossip Coherence Simulation...")
    # Simular 100 ciclos de gossip (simplificado)
    coherence_history = []
    for cycle in range(100):
        # A cada ciclo, os embeddings podem ser ligeiramente ajustados pelo protocolo
        # (aqui mantemos fixos para simplicidade)
        distances = []
        names = list(embeddings.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                distances.append(d_h(embeddings[names[i]], embeddings[names[j]]))
        c = np.exp(-np.mean(distances) * 0.1)  # fator de escala calibrado
        coherence_history.append(c)

    # Resultados
    print(f"Coerência inicial: {coherence_history[0]:.4f}")
    print(f"Coerência após estabilização: {coherence_history[-1]:.4f}")
    print(f"Média final: {np.mean(coherence_history[-10:]):.4f}")
    return coherence_history

if __name__ == "__main__":
    run_sim()
