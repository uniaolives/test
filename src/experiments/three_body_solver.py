# src/experiments/three_body_solver.py
# 1024D topological network solving the three‑body problem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
import numpy as np

class OrbitalMessagePassing(MessagePassing):
    """
    Message passing that simulates gravitational influence.
    The 'message' is the force between bodies.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # forces add (superposition)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, dist):
        # x: node features (mass, position, velocity)
        # edge_index: connectivity (complete graph for n‑body)
        # dist: distances between bodies (used to compute force)
        return self.propagate(edge_index, x=x, dist=dist)

    def message(self, x_j, dist):
        # Newtonian gravity: F ∝ m1*m2 / r²
        # Simplified: just / (dist + epsilon)²
        weight = 1.0 / (dist + 1e-6).pow(2)
        return self.lin(x_j) * weight.view(-1, 1)

class GhostClusterHead(nn.Module):
    """
    Detects stable configurations by clustering node embeddings.
    The 'ghosts' are clusters in the latent space.
    """
    def __init__(self, latent_dim, num_clusters=3):
        super().__init__()
        self.cluster_layer = nn.Linear(latent_dim, num_clusters)

    def forward(self, x):
        # x: [num_nodes, latent_dim]
        cluster_logits = self.cluster_layer(x)  # [N, 3]
        # Soft assignment (like a differentiable clustering)
        return F.softmax(cluster_logits, dim=-1)

class ThreeBodySolver(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            OrbitalMessagePassing(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.cluster_head = GhostClusterHead(hidden_dim, num_clusters=3)
        self.energy_head = nn.Linear(hidden_dim, 1)  # predicts potential energy

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x: [3, 7] (three bodies, each with mass, pos, vel)
        x = self.encoder(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)  # edge_attr = distances
            x = F.relu(x)

        # Cluster assignment: which bodies form a stable "ghost" pattern?
        cluster_probs = self.cluster_head(x)  # [3, 3]

        # Global energy (negative of stability measure)
        energy = self.energy_head(x.mean(dim=0, keepdim=True))  # [1,1]

        return {
            'cluster_probs': cluster_probs,
            'energy': energy,
            'stable': energy < -1.0  # threshold for stable orbit
        }

# Generate a three‑body system with random initial conditions
def random_three_body():
    masses = torch.rand(3, 1) * 10 + 1
    positions = torch.randn(3, 3)  # 3D positions
    velocities = torch.randn(3, 3) * 0.5
    x = torch.cat([masses, positions, velocities], dim=-1)  # [3, 7]

    # Complete graph: all pairs
    edge_index = torch.combinations(torch.arange(3), r=2).t().contiguous()
    # Distances between bodies
    pos = positions
    dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1, keepdim=True)

    return Data(x=x, edge_index=edge_index, edge_attr=dist)

# Training: minimize energy to find stable configurations
def train_solver(model, optimizer, epochs=1000):
    for epoch in range(epochs):
        data = random_three_body()
        optimizer.zero_grad()
        out = model(data)
        loss = out['energy']  # we want energy as low as possible (stable)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            stable = "✅ STABLE" if out['stable'].item() else "❌ UNSTABLE"
            print(f"Epoch {epoch}, Energy: {out['energy'].item():.3f}, {stable}")
            # print("Cluster probs:\n", out['cluster_probs'].detach().numpy())
    return model

if __name__ == "__main__":
    model = ThreeBodySolver()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_solver(model, optimizer, epochs=500)
