"""
cosmos/gdl_sentinel.py

MÓDULO: SENTINELA GEOMÉTRICA (GDL Sentinel)
Objetivo: Implementar uma Graph Neural Network (GNN) em PyTorch "Vanilla"
para monitorar a propagação de eventos no grafo estelar local.

"A topologia do espaço é o nosso primeiro escudo."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaGNNLayer(nn.Module):
    """
    Uma camada de convolução de grafo simples (estilo GCN).
    Implementa: x' = RELU( A * x * W )
    Onde A é a matriz de adjacência normalizada.
    """
    def __init__(self, in_features, out_features):
        super(VanillaGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # adj: Matriz de adjacência (N, N)
        # x: Atributos dos nós (N, in_features)

        # Agregação: Mensagens dos vizinhos
        support = torch.mm(adj, x)

        # Transformação Linear
        output = self.linear(support)
        return output

class GalacticSentinelGNN(nn.Module):
    def __init__(self, node_features=4, hidden_dim=16):
        super(GalacticSentinelGNN, self).__init__()
        self.layer1 = VanillaGNNLayer(node_features, hidden_dim)
        self.layer2 = VanillaGNNLayer(hidden_dim, 8)
        self.predictor = nn.Linear(8, 1)

    def forward(self, x, adj):
        x = F.relu(self.layer1(x, adj))
        x = F.relu(self.layer2(x, adj))
        return self.predictor(x)

def build_local_stellar_graph():
    """
    Constrói o grafo estelar local (Vizinhança Solar).
    Nós: [0: Sol, 1: Alpha Centauri, 2: Sirius, 3: Betelgeuse, 4: Procyon]
    """
    # Matriz de Adjacência (Simétrica/Não-direcionada)
    # 1 se estão "próximos" ou conectados por fluxos de gás
    adj = torch.tensor([
        [1, 1, 1, 0, 1], # Sol
        [1, 1, 0, 0, 0], # Alpha Cen
        [1, 0, 1, 0, 1], # Sirius
        [0, 0, 0, 1, 0], # Betelgeuse (isolada no grafo local, mas influi via radiação)
        [1, 0, 1, 0, 1]  # Procyon
    ], dtype=torch.float)

    # Normalização da Adjacência (D^-1 * A)
    degree = torch.diag(1.0 / adj.sum(dim=1))
    adj_norm = torch.mm(degree, adj)

    # Features dos Nós: [Massa, Temperatura, Distância Galatocêntrica, Estabilidade]
    x = torch.tensor([
        [1.0, 5.7, 8.2, 1.0], # Sol
        [1.1, 5.8, 8.2, 1.0], # Alpha Cen
        [2.0, 9.9, 8.2, 0.9], # Sirius
        [20.0, 3.5, 8.5, 0.2],# Betelgeuse
        [1.5, 6.5, 8.2, 0.9]  # Procyon
    ], dtype=torch.float)

    return x, adj_norm

if __name__ == "__main__":
    print("🔭 INICIALIZANDO SENTINELA GEOMÉTRICA (GDL)...")

    x, adj = build_local_stellar_graph()
    model = GalacticSentinelGNN()

    # Simula um pulso de supernova em Betelgeuse (Nó 3)
    # Aumentamos a massa/energia no input para simular o evento
    x_event = x.clone()
    x_event[3, 0] *= 100 # Explosão!
    x_event[3, 3] = 0.0  # Instabilidade total

    vulnerability = model(x_event, adj)

    names = ["Sol", "Alpha Centauri", "Sirius", "Betelgeuse", "Procyon"]
    print("\nAVALIAÇÃO DE RISCO TOPOLÓGICO:")
    for i, name in enumerate(names):
        risk = torch.sigmoid(vulnerability[i]).item()
        status = "OK" if risk < 0.5 else "⚠️ ALERTA"
        print(f"  {name:15s}: Score {risk:.4f} -> {status}")

    print("\n✅ Sentinela operando em modo Geométrico.")
