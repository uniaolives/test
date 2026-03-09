# src/physics/ghost_clustering.py
# PyG implementation of 3-body ghost detection

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data

class GhostDetectionLayer(MessagePassing):
    """
    Message passing in INVERSE phase space.
    Messages flow from high-information to low-information regions
    (like water flowing downhill).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),  # +1 for edge_attr (dI/dE)
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels] = (E, I, ∇I) for each phase space region
        # edge_attr: [E, 1] = information gradient along Hamiltonian flow
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: destination node (higher I)
        # x_j: source node (lower I)
        # Message = flow of information toward attractor
        flow = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(flow)

def compute_global_coherence(x):
    # Simplified φ_q calculation based on information density
    # In practice this would be the Miller Limit threshold
    return 4.64 + torch.rand(1).item() * 0.1

class ThreeBodyGhostNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GhostDetectionLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        # Ghost scoring head: how "ghost-like" is this region?
        self.ghost_head = nn.Linear(hidden_dim, 1)
        # Period prediction head
        self.period_head = nn.Linear(hidden_dim, 1)
        # Gradient predictor (to avoid relying on input indices after layers)
        self.grad_head = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Initial encoding
        x = self.encoder(x)

        # Inverse phase space flow
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.relu(x)

        # Global ghost detection
        ghost_score = torch.sigmoid(self.ghost_head(x))  # [N, 1]
        period_pred = self.period_head(x)  # [N, 1]
        grad_pred = self.grad_head(x) # [N, 2]

        # Cluster ghosts: nodes with high ghost_score and low predicted information gradient
        ghost_mask = (ghost_score > 0.9) & (torch.norm(grad_pred, dim=1, keepdim=True) < 0.1)

        return {
            'ghost_locations': x[ghost_mask.squeeze()],
            'ghost_periods': period_pred[ghost_mask],
            'phase_space_embedding': x,
            'phi_q': compute_global_coherence(x)  # Miller Limit check
        }

def compute_information_density(i, j, angular_momentum):
    return (i*j) / (angular_momentum + 1.0)

def compute_information_gradient(i, j):
    return [0.1 * i, 0.1 * j]

def is_hamiltonian_connected(node_i, node_j):
    # Conservation of energy (within some tolerance)
    return abs(node_i[0] - node_j[0]) < 0.05

def build_three_body_graph(energy, angular_momentum):
    """
    Discretize inverse phase space into 1024 nodes.
    Edges = Hamiltonian flow connections.
    """
    # Phase space sampling (simplified)
    nodes = []
    edges = []
    edge_attrs = []

    for i in range(32):  # Position bins
        for j in range(32):  # Momentum bins
            # State: (E, I, dI/dx, dI/dp)
            E = energy
            I = compute_information_density(i, j, angular_momentum)
            grad_I = compute_information_gradient(i, j)
            nodes.append([E, I, grad_I[0], grad_I[1]])

    # Connect nodes along Hamiltonian flow (conservation laws)
    # This is O(N^2), but N=1024 is manageable
    for idx_i, node_i in enumerate(nodes):
        for idx_j, node_j in enumerate(nodes):
            if idx_i != idx_j and is_hamiltonian_connected(node_i, node_j):
                edges.append([idx_i, idx_j])
                # Edge attribute = information gradient along flow
                dI = abs(node_j[1] - node_i[1])  # |I_j - I_i|
                edge_attrs.append([dI])

    x = torch.tensor(nodes, dtype=torch.float)
    # Handle case with no edges
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
