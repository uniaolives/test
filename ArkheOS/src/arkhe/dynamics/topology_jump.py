"""
First Quantum Jump in Network Topology
Simulating spontaneous structural evolution
"""

import numpy as np
import matplotlib.pyplot as plt

class TopologyJump:
    """
    Quantum jump in hypergraph topology

    Smolin: Networks evolve through discrete topology changes
    """

    def __init__(self, initial_nodes: int = 5):
        self.nodes = initial_nodes
        self.edges = []
        self.time = 0
        self.history = []

        # Initial random topology
        for i in range(initial_nodes):
            for j in range(i+1, initial_nodes):
                if np.random.random() > 0.5:
                    self.edges.append((i, j))

        self.record_state()

    def record_state(self):
        """Record topology state"""
        self.history.append({
            'time': self.time,
            'nodes': self.nodes,
            'edges': len(self.edges),
            'topology': 'connected' if self.is_connected() else 'fragmented'
        })

    def is_connected(self) -> bool:
        """Check if graph is connected"""
        if self.nodes == 0:
            return True

        visited = set([0])
        queue = [0]

        while queue:
            current = queue.pop(0)
            for i, j in self.edges:
                if i == current and j not in visited:
                    visited.add(j)
                    queue.append(j)
                elif j == current and i not in visited:
                    visited.add(i)
                    queue.append(i)

        return len(visited) == self.nodes

    def quantum_jump(self):
        """
        Execute quantum topology jump

        Possible jumps:
        - Add node (universe expansion)
        - Remove node (black hole evaporation)
        - Add edge (entanglement creation)
        - Remove edge (decoherence)
        """
        self.time += 1

        jump_type = np.random.choice(['add_node', 'remove_node', 'add_edge', 'remove_edge'])

        if jump_type == 'add_node':
            # Add new node connected to random existing
            new_node = self.nodes
            self.nodes += 1

            # Connect to 1-3 random nodes
            n_connections = np.random.randint(1, min(4, self.nodes))
            for _ in range(n_connections):
                target = np.random.randint(0, new_node)
                self.edges.append((target, new_node))

            return f"Added node {new_node}"

        elif jump_type == 'remove_node' and self.nodes > 2:
            # Remove random node (if >= 2 nodes remain)
            victim = np.random.randint(0, self.nodes)

            # Remove edges involving victim
            self.edges = [(i, j) for i, j in self.edges if i != victim and j != victim]

            # Relabel higher nodes
            self.edges = [(i if i < victim else i-1, j if j < victim else j-1)
                         for i, j in self.edges]

            self.nodes -= 1

            return f"Removed node {victim}"

        elif jump_type == 'add_edge':
            # Add edge between unconnected nodes
            possible = [(i, j) for i in range(self.nodes)
                       for j in range(i+1, self.nodes)
                       if (i, j) not in self.edges]

            if possible:
                new_edge = possible[np.random.randint(0, len(possible))]
                self.edges.append(new_edge)
                return f"Added edge {new_edge}"

        elif jump_type == 'remove_edge' and len(self.edges) > 0:
            # Remove random edge
            victim = self.edges[np.random.randint(0, len(self.edges))]
            self.edges.remove(victim)
            return f"Removed edge {victim}"

        return "No change"

    def simulate_evolution(self, steps: int = 20):
        """Simulate topology evolution"""

        print(f"\n⚡ Simulating Topology Evolution ({steps} quantum jumps)...")
        print()

        print(f"Initial state:")
        print(f"  Nodes: {self.nodes}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  Connected: {self.is_connected()}")
        print()

        for step in range(steps):
            action = self.quantum_jump()
            self.record_state()

            if step % 5 == 0:
                state = self.history[-1]
                print(f"Step {step}: {action}")
                print(f"  Nodes: {state['nodes']}, Edges: {state['edges']}, {state['topology']}")

        final = self.history[-1]
        print()
        print(f"Final state:")
        print(f"  Nodes: {final['nodes']}")
        print(f"  Edges: {final['edges']}")
        print(f"  Connected: {final['topology']}")
        print()
        print("✅ Topology evolved through quantum jumps")

def demonstrate_topology_jump():
    """Demonstrate quantum topology evolution"""

    print("="*70)
    print("QUANTUM TOPOLOGY JUMP: NETWORK EVOLUTION")
    print("="*70)

    jump = TopologyJump(initial_nodes=5)
    jump.simulate_evolution(steps=20)

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    print("Quantum jumps in topology represent:")
    print("  • Universe expansion/contraction (add/remove nodes)")
    print("  • Entanglement creation/decoherence (add/remove edges)")
    print("  • Discrete evolution (not continuous)")
    print("  • Natural selection (connected topologies survive)")
    print()
    print("Arkhe(N) topology is not fixed—it evolves")
    print("through quantum jumps, just like space-time itself.")
    print()
    print("∞")

    return jump

if __name__ == "__main__":
    jump = demonstrate_topology_jump()
