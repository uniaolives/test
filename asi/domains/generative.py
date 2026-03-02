import numpy as np
from core.hypergraph import Hypergraph

def simulate_latent_forcing(h: Hypergraph, num_latents=16, num_pixels=256):
    """Latent forcing diffusion: generate latents first, then pixels."""
    for i in range(num_latents):
        h.add_node(data={"type": "latent", "index": i, "order": "first"})
    for i in range(num_pixels):
        h.add_node(data={"type": "pixel", "index": i, "order": "second"})

    # Connect each latent to a subset of pixels (simulating conditioning)
    latents = [nid for nid, n in h.nodes.items() if n.data.get("type") == "latent"]
    pixels = [nid for nid, n in h.nodes.items() if n.data.get("type") == "pixel"]
    for i, l in enumerate(latents):
        # Each latent influences a random subset of pixels
        subset = np.random.choice(pixels, size=int(len(pixels)*0.2), replace=False)
        for p in subset:
            h.add_edge({l, p}, weight=np.random.uniform(0.5, 1.0))
    h.bootstrap_step()
