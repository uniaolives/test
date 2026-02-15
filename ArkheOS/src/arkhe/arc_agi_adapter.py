import json
import numpy as np
from .arkhe_agi import PyAGICore
from .ontological_memory import find_similar_concepts

class ARCAdapter:
    """
    Adapter for the ARC-AGI benchmark.
    """
    def __init__(self, core: PyAGICore):
        self.core = core
        self.results = []

    def grid_to_embedding(self, grid):
        """Converts 2D grid to embedding (flat + normalized)."""
        flat = np.array(grid).flatten()
        return (flat / 9.0).tolist()  # values 0-9

    def solve_task(self, task):
        """Resolves an ARC task using the AGI Core."""
        train_pairs = task['train']
        test_input = task['test'][0]['input']

        # Feed the AGI with training examples
        for ex in train_pairs:
            emb_in = self.grid_to_embedding(ex['input'])
            emb_out = self.grid_to_embedding(ex['output'])
            self.core.add_node(int(np.random.rand()*1e6), 0, 0, emb_in)
            self.core.add_node(int(np.random.rand()*1e6), 0, 0, emb_out)
            self.core.handover_step(0.01, 0.01)

        # Generate test embedding and query ontology
        emb_test = self.grid_to_embedding(test_input)
        similar = find_similar_concepts(emb_test, top_k=3)

        # Final handover to produce output
        for _ in range(5):
            self.core.handover_step(0.02, 0.03)
        output_embedding = self.core.get_last_node_embedding()

        # Simplified reconstruction for demo
        return test_input # Placeholder

    def run_benchmark(self, tasks_file):
        with open(tasks_file, 'r') as f:
            tasks = json.load(f)
        correct = 0
        for i, task in enumerate(tasks):
            output = self.solve_task(task)
            expected = task['test'][0]['output']
            if output == expected:
                correct += 1
            self.results.append((i, output, expected))
        score = correct / len(tasks) if tasks else 0
        print(f"ARC-AGI Score: {score*100:.2f}%")
        return score
