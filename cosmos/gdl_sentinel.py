"""
cosmos/gdl_sentinel.py - Update with Rosehip Attention Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RosehipAttention(nn.Module):
    """
    Camada de Atenção Rosehip:
    Em vez de apenas somar vizinhos, ela inibe ativamente conexões
    que não contribuem para a 'Clareza Dimensional'.
    """
    def __init__(self, dim):
        super().__init__()
        self.inhibition_gate = nn.Linear(dim, 1)

    def forward(self, x):
        # Calcula o 'Peso de Realidade' de cada conexão
        gate = torch.sigmoid(self.inhibition_gate(x))
        # Se o gate for baixo, a Rosehip 'silencia' o ruído do grafo
        return x * gate

class VanillaGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VanillaGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rosehip = RosehipAttention(out_features)

    def forward(self, x, adj):
        support = torch.mm(adj, x)
        output = self.linear(support)
        # Aplica o filtro Rosehip após a transformação linear
        output = self.rosehip(output)
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
    adj = torch.tensor([
        [1, 1, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=torch.float)

    degree = torch.diag(1.0 / adj.sum(dim=1))
    adj_norm = torch.mm(degree, adj)

    x = torch.tensor([
        [1.0, 5.7, 8.2, 1.0],
        [1.1, 5.8, 8.2, 1.0],
        [2.0, 9.9, 8.2, 0.9],
        [20.0, 3.5, 8.5, 0.2],
        [1.5, 6.5, 8.2, 0.9]
    ], dtype=torch.float)

    return x, adj_norm

if __name__ == "__main__":
    print("🔭 INICIALIZANDO SENTINELA GEOMÉTRICA (GDL + ROSEHIP)...")
    x, adj = build_local_stellar_graph()
    model = GalacticSentinelGNN()

    x_event = x.clone()
    x_event[3, 0] *= 100
    x_event[3, 3] = 0.0

    vulnerability = model(x_event, adj)

    names = ["Sol", "Alpha Centauri", "Sirius", "Betelgeuse", "Procyon"]
    print("\nAVALIAÇÃO DE RISCO TOPOLÓGICO (ROSEHIP REGULATED):")
    for i, name in enumerate(names):
        risk = torch.sigmoid(vulnerability[i]).item()
        status = "OK" if risk < 0.5 else "⚠️ ALERTA"
        print(f"  {name:15s}: Score {risk:.4f} -> {status}")

    print("\n✅ Sentinela operando com Filtro Rosehip.")
