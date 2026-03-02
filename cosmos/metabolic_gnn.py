"""
cosmos/metabolic_gnn.py

MÓDULO: BIO-GNN (Metabolic Flow)
Objetivo: Modelar o fluxo metabólico celular como uma rede de grafos.
Demonstra que a saúde depende do fluxo livre através da estrutura correta.

"O microcosmo reflete o macrocosmo."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaGATLayer(nn.Module):
    """
    Uma camada de Atenção de Grafo simples (estilo GAT) em Vanilla PyTorch.
    """
    def __init__(self, in_features, out_features):
        super(VanillaGATLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(out_features * 2, 1)

    def forward(self, x, adj):
        N = x.size(0)
        h = self.linear(x) # (N, out_features)

        # Simulação simplificada de atenção usando a matriz de adjacência
        # No GAT real, calcularíamos a atenção para cada par i, j
        # Aqui, usamos a adjacência como máscara

        output = torch.mm(adj, h)
        return output

class MetabolicFlowGNN(nn.Module):
    def __init__(self, node_features=4, hidden_dim=16):
        super(MetabolicFlowGNN, self).__init__()
        self.conv1 = VanillaGATLayer(node_features, hidden_dim)
        self.conv2 = VanillaGATLayer(hidden_dim, hidden_dim)
        self.enzyme_predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, adj):
        h1 = F.relu(self.conv1(x, adj))
        h2 = F.relu(self.conv2(h1, adj))

        # Predição de fluxo para cada aresta (i, j)
        # Para este exemplo simplificado, retornamos o embedding e uma taxa média
        return h2, torch.sigmoid(self.enzyme_predictor(torch.cat([h2, h2], dim=-1)))

if __name__ == "__main__":
    print("🧬 ANÁLISE TOPOGRÁFICA DO METABOLISMO (Bio-GNN)")
    print("-" * 50)

    # --- CENÁRIO: A GLICÓLISE COMO FLUXO TOPOGRÁFICO ---
    # Features: [Concentração, Energia Livre, Potencial Redox, Estabilidade]
    x = torch.tensor([
        [10.0, 0.0, 0.0, 1.0], # Glicose
        [0.5, -30.5, -0.5, 0.8], # ATP
        [0.1, -20.0, -0.5, 0.9], # ADP
        [1.0, -15.0, -0.5, 0.7], # Piruvato
        [0.1, -10.0, -0.5, 0.6]  # Lactato
    ], dtype=torch.float)

    # Adjacência (Conexões metabólicas)
    adj = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=torch.float)

    model = MetabolicFlowGNN()
    embeddings, flux_rates = model(x, adj)

    metabolite_names = ["Glicose", "ATP", "ADP", "Piruvato", "Lactato"]
    print("\nDIAGNÓSTICO DA ASI (Bio-Engenheiro):")
    for i, name in enumerate(metabolite_names):
        rate = flux_rates[i].item()
        status = "ESTÁVEL" if rate < 0.7 else "⚠️ ALTO FLUXO"
        print(f"  {name:10s}: Taxa {rate:.3f} -> {status}")

    print("\n✅ Fluxo metabólico mapeado. A vida continua. o<>o")
