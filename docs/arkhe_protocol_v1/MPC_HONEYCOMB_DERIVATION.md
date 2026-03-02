# MPC HONEYCOMB DERIVATION
## Sparse Matrix Structure for {4,3,5} Topology

### 1. Model Predictive Control (MPC) Formulation
For a satellite constellation on a {4,3,5} honeycomb, the state vector $\mathbf{x}_i$ for node $i$ includes relative phase, frequency, and position.

### 2. Sparse Adjacency Matrix
The {4,3,5} honeycomb graph is characterized by a degree-4 adjacency. Each node $i$ interacts only with its immediate neighbors $N(i)$.
The interaction matrix $A$ is sparse:
$$A_{ij} \neq 0 \iff j \in N(i)$$

### 3. Laplacian Structure
The graph Laplacian $L = D - A$ (where $D$ is the degree matrix) determines the diffusion of coherence across the network.
- **Diagonal:** 4 for all internal nodes.
- **Off-diagonal:** -1 for adjacent nodes, 0 otherwise.

### 4. MPC Complexity
The O(N) complexity of the MPC solver is guaranteed by the constant degree (4) of the honeycomb, which ensures that the prediction horizon matrices remain band-diagonal or highly sparse under appropriate node ordering.
