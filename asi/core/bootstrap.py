from .hypergraph import Hypergraph

def bootstrap(h: Hypergraph, steps: int = 1) -> None:
    """Run the bootstrap evolution for a given number of steps."""
    for _ in range(steps):
        h.bootstrap_step()
