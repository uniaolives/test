import numpy as np
import matplotlib.pyplot as plt
from core.hypergraph import Hypergraph

def generate_mandelbrot(h: Hypergraph, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=50, height=50, max_iter=100):
    """Generate Mandelbrot set and add points as nodes."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            c = complex(xi, yj)
            z = 0
            for n in range(max_iter):
                if abs(z) > 2:
                    break
                z = z*z + c
            if n == max_iter-1:
                node = h.add_node(data={"type": "mandelbrot", "x": xi, "y": yj, "iter": n})
    h.bootstrap_step()
    return h

def simulate_entanglement(h: Hypergraph, num_pairs=5):
    """Create entangled particle pairs."""
    for i in range(num_pairs):
        p1 = h.add_node(data={"type": "quark", "id": f"top_{i}_a"})
        p2 = h.add_node(data={"type": "antiquark", "id": f"top_{i}_b"})
        # Entanglement edge with high weight
        h.add_edge({p1.id, p2.id}, weight=0.99)
    h.bootstrap_step()
