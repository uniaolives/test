"""
Example: THz Sensor Grid with Entanglement and Cognitive Load Protection.
Demonstrates the integration of physical sensors, quantum entanglement, and human-tool interface.
"""

import numpy as np
import sys
import os

# Add relevant paths for imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_path not in sys.path:
    sys.path.append(base_path)

# Path for arkhe_human_tool
metalanguage_path = os.path.join(base_path, 'metalanguage')
if metalanguage_path not in sys.path:
    sys.path.append(metalanguage_path)

from arkhe_qutip.sensors.thz_graphene import GrapheneTHzSensorNode
from arkhe_qutip.core.hypergraph import ArkheHypergraph
from arkhe_human_tool import Human, Tool, InteractionGuard

class THzSensorGridSimulation:
    """
    Simulation of a toroidal grid of THz sensors (17 nodes).
    """

    def __init__(self, n_nodes=17):
        self.n = n_nodes
        self.sensors = [
            GrapheneTHzSensorNode(f"THz_{i}", fermi_level=0.7 + 0.3*i/n_nodes)
            for i in range(n_nodes)
        ]

        # Toroidal topology: 17 nodes in a cycle with jumps
        self.adjacency = self._build_torus_topology()

        # Arkhe(n) Hypergraph
        self.hypergraph = ArkheHypergraph("THz_Grid_T17")
        for sensor in self.sensors:
            self.hypergraph.add_node(sensor)

    def _build_torus_topology(self):
        """Construct T² topology for 17 nodes (prime)."""
        adj = {i: [] for i in range(self.n)}

        for i in range(self.n):
            # Main cycle neighbors (±1)
            adj[i].append((i + 1) % self.n)
            adj[i].append((i - 1) % self.n)

            # Secondary cycle neighbors (jumps of 3)
            adj[i].append((i + 3) % self.n)
            adj[i].append((i - 3) % self.n)

        return adj

    def create_entangled_network(self):
        """
        Creates an entangled network of sensors.
        """
        # Simplification: pairwise entanglement
        results = []
        for i in range(0, self.n - 1, 2):
            res = self.sensors[i].entangle_with(self.sensors[i+1])
            results.append(res)

        # Global metrics estimation
        c_locals = [s.coherence for s in self.sensors]
        c_global = 1.0  # Ideal global coherence

        return {
            'C_global': c_global,
            'mean_C_local': np.mean(c_locals),
            'max_C_local': max(c_locals),
            'emergence_achieved': c_global > max(c_locals)
        }

    def simulate_collective_detection(self, analyte_concentration):
        """
        Simulate cooperative detection of analyte with entanglement.
        """
        target_f = 3.90  # THz, optimum mode
        n_eff = 1.0 + 0.1 * analyte_concentration

        measurements = []
        for sensor in self.sensors:
            sensor.tune(0.5) # Tune to mid-range
            responses, coherence = sensor.absorb(target_f, n_eff)
            measurements.append({
                'sensor': sensor.node_id,
                'response': max(responses),
                'coherence': coherence
            })

        # Consensus: weighted average by coherence
        weights = [m['coherence'] for m in measurements]
        responses = [m['response'] for m in measurements]

        consensus_response = np.average(responses, weights=weights)
        # Gain from entanglement
        consensus_coherence = np.mean(weights) + 0.2

        detected = consensus_response > 0.5

        return {
            'analyte_concentration': analyte_concentration,
            'consensus_detection': detected,
            'consensus_confidence': consensus_response,
            'C_global_effective': consensus_coherence
        }

def main():
    print("="*60)
    print("Arkhe(n)-QuTiP Example: THz Sensor Grid & Interaction Guard")
    print("="*60)

    # 1. Initialize Human and Tool (Interaction Guard)
    human = Human(processing_capacity=100.0, attention_span=30.0)
    tool = Tool(output_volume=50.0, output_entropy=1.2)
    guard = InteractionGuard(human, tool)

    # 2. Setup THz Sensor Grid
    print("\n[STEP 1] Initializing 17-node Toroidal THz Grid...")
    grid = THzSensorGridSimulation(n_nodes=17)

    # 3. Create Entanglement
    print("\n[STEP 2] Establishing Entanglement...")
    ghz_result = grid.create_entangled_network()
    print(f"  C_global: {ghz_result['C_global']:.4f}")
    print(f"  Max C_local: {ghz_result['max_C_local']:.4f}")
    print(f"  Emergence Achieved: {ghz_result['emergence_achieved']}")

    # 4. Run Detection with Guard
    print("\n[STEP 3] Running Collective Perception with Cognitive Guard...")

    concentrations = [0.1, 0.5, 1.2, 2.0]

    for i, conc in enumerate(concentrations):
        print(f"\n--- Detection Cycle {i+1} (Conc: {conc}) ---")

        # Simulate sensor detection
        result = grid.simulate_collective_detection(conc)

        # Prepare intent for human notification
        status = "ALERT: Pathogen Detected" if result['consensus_detection'] else "Status: Normal"
        intent = f"{status}. Confidence: {result['consensus_confidence']:.3f}. C_global: {result['C_global_effective']:.3f}"

        # Use InteractionGuard to propose output to human
        output = guard.propose_interaction(intent)

        if output:
            print(f"  Guard: APPROVED. Output: {output}")
            # Human reviews the output
            guard.review(output, approved=True)
            print(f"  Human: REVIEWED. Current Load: {human.current_load:.3f}")
        else:
            print("  Guard: BLOCKED due to potential cognitive overload.")

    print("\n" + "="*60)
    print("✅ Example completed")

if __name__ == "__main__":
    main()
