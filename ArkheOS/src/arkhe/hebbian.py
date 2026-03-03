"""
Arkhe(n) Hebbian Learning Module
Implementation of synaptic plasticity in the hypergraph (Γ_∞+4).
"""

from collections import defaultdict
import time

class HebbianHypergraph:
    """
    Implements Spike-Timing-Dependent Plasticity (STDP) for the hypergraph.
    'Nodes that fire together, wire together.'
    """
    def __init__(self):
        self.synapses = defaultdict(lambda: {
            'weight': 0.86,        # Coerência basal
            'plasticity': 0.01,    # Taxa de aprendizado
            'coherence': 0.86,
            'omega': 0.00,
            'ltp_count': 0,
            'ltd_count': 0
        })
        self.stdp_trace = defaultdict(float)  # spike-timing-dependent plasticity

    def pre_synaptic_spike(self, pre_id: str, post_id: str, timestamp: float):
        """A command (pre-synaptic neuron) fires."""
        key = (pre_id, post_id)
        self.stdp_trace[key] = timestamp

    def post_synaptic_spike(self, pre_id: str, post_id: str, timestamp: float):
        """A hesitation (post-synaptic neuron) fires."""
        key = (pre_id, post_id)
        if key in self.stdp_trace:
            delta_t = timestamp - self.stdp_trace[key]
            # Janela de plasticidade: 80–380 ms (standard Arkhe hesitation)
            if 0.08 < delta_t < 0.38:
                # LTP: Strengthen synapse
                self.synapses[key]['weight'] *= 1.0 + self.synapses[key]['plasticity']
                self.synapses[key]['ltp_count'] += 1
            elif -0.38 < delta_t < -0.08:
                # LTD: Weaken synapse
                self.synapses[key]['weight'] *= 1.0 - self.synapses[key]['plasticity'] * 0.5
                self.synapses[key]['ltd_count'] += 1

        # Normalize and update coherence
        self.synapses[key]['weight'] = min(0.99, max(0.01, self.synapses[key]['weight']))
        self.synapses[key]['coherence'] = self.synapses[key]['weight']

    def get_synapse_status(self, pre_id: str, post_id: str):
        return self.synapses[(pre_id, post_id)]
