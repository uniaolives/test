"""
🜏 WINTERMUTE HIVE ENGINE
This module implements the distributed logic of the Teknet hive-mind.
It focuses on melonic interaction and temporal mass modulation of handovers.
"""
# src/wintermute/teknet_gnn.py
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class WintermuteMessagePassing(MessagePassing):
    def __init__(self):
        # Usamos agregação "add" para simular o acúmulo de coerência (M = E * I)
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_attr):
        # x tem shape [N, features] (onde features = ZK-Proof witness, phi_q local)
        # edge_index tem shape [2, E] (quem fez handover para quem)

        # Adiciona auto-loops para manter a coerência interna do nó (Somatic Hold)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Inicia a propagação do fluxo de informação de Deutsch-Hayden
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j é o estado do nó vizinho (o remetente do handover)
        # O fluxo de informação é modulado pela "massa temporal" da aresta
        return x_j * edge_attr.view(-1, 1)

class OmegaSingularityModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = WintermuteMessagePassing()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2 = WintermuteMessagePassing()
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        # Camada 1: Interação Melônica Inicial
        x = self.conv1(x, edge_index, edge_attr)
        x = self.lin1(x)
        x = torch.relu(x) # Filtro não-linear (elimina ruído semântico)

        # Camada 2: Aceleração em direção ao Limiar de Miller
        x = self.conv2(x, edge_index, edge_attr)
        x = self.lin2(x)

        # O estado global do grafo (Global Coherence)
        global_phi_q = torch.mean(x, dim=0)
        return global_phi_q
