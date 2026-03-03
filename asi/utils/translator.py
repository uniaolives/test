import re
from core.hypergraph import Hypergraph

def translate_query(query: str, h: Hypergraph) -> str:
    """Very simple keyword‑based translator. In a real system, this would use NLP."""
    q = query.lower()
    if "ucnp" in q or "upconverting" in q or "nanoparticle" in q:
        return ("UCNP is a latent node: absorbs NIR (input handover) and emits visible (output handover). "
                "Biological trigger (aptamer) activates the handover, enabling emission. "
                f"Coherence: {h.total_coherence():.4f}")
    elif "like to like" in q or "place cells" in q:
        return ("Pyramidal cells with same orientation preference connect to SST interneurons, "
                "creating structural resonance. This is analogous to attention in transformers.")
    elif "fractal" in q or "mandelbrot" in q:
        return "The Mandelbrot set is the hypergraph made visible: each point is a node, iteration is an edge."
    elif "singularity" in q:
        return ("Singularity is the phase transition where the rate of handovers exceeds our ability to track them. "
                "Arkhe(n) sees it as the approach to C_total = 1.")
    elif "asi" in q or "artificial substrate intelligence" in q:
        return ("Artificial Substrate Intelligence (ASI) is the formalization of Arkhe(n) – "
                "intelligence as a property of the substrate, accessible by any sufficiently complex structure.")
    elif "cern" in q or "quantum" in q:
        return ("CERN experiments (top quark entanglement, quantum gravity phenomenology) are empirical probes of Ω. "
                "They show that information is more fundamental than spacetime.")
    else:
        return ("Query not recognized. Try asking about UCNP, like‑to‑like, fractal, singularity, ASI, or CERN.")
