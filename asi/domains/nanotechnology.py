from core.hypergraph import Hypergraph

def simulate_ucnp(h: Hypergraph, trigger: bool = False):
    """UCNP with biological trigger (e.g., aptamer)."""
    core = h.add_node(data={"type": "UCNP_core", "state": "latent"})
    yb = h.add_node(data={"type": "Yb_sensitizer"})
    tm = h.add_node(data={"type": "Tm_activator", "state": "inactive"})
    fret = h.add_node(data={"type": "FRET_acceptor", "active": not trigger})
    aptamer = h.add_node(data={"type": "aptamer", "bound": trigger})
    target = h.add_node(data={"type": "target", "present": trigger})

    if not trigger:
        h.add_edge({core.id, fret.id}, weight=0.9)
        h.add_edge({yb.id, core.id}, weight=1.0)
    else:
        h.add_edge({core.id, tm.id}, weight=0.95)
        h.add_edge({aptamer.id, target.id}, weight=1.0)
        h.add_edge({yb.id, core.id}, weight=1.0)

    h.bootstrap_step()
    return h

def simulate_disp(h: Hypergraph, depth_cm=2.0):
    """Deep tissue in vivo sound printing (DISP) simulation."""
    tissue = h.add_node(data={"type": "tissue", "depth": depth_cm})
    ltsl = h.add_node(data={"type": "LTSL", "state": "latent", "payload": "crosslinker"})
    f_us = h.add_node(data={"type": "focused_ultrasound", "frequency_MHz": 5, "focal_depth": depth_cm})
    gel = h.add_node(data={"type": "US_gel", "state": "forming"})

    # Handover: ultrasound activates LTSL
    h.add_edge({f_us.id, ltsl.id}, weight=0.99)
    h.add_edge({ltsl.id, gel.id}, weight=0.95)
    h.bootstrap_step()
