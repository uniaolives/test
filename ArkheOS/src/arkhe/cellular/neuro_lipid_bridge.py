"""
Neuro-Lipid Bridge
Connecting cellular lipid signaling to neural hierarchical coding
"""

class IonChannel:
    def __init__(self, channel_type: str, bound_pi: str):
        self.type = channel_type
        self.bound_pi = bound_pi
        self.open_probability = 0.5

class NeuroLipidInterface:
    def __init__(self, lipid_cell, hdc_brain):
        """
        lipid_cell: Object with 'ion_channels' list
        hdc_brain: BrainHypergraph instance
        """
        self.cell = lipid_cell
        self.brain = hdc_brain

    def propagate_signal(self):
        # PI(4,5)P2 ativa canais de potássio → modula potencial de membrana
        print("🔗 Neuro-Lipid Bridge: Propagating signal...")

        if hasattr(self.cell, 'ion_channels'):
            k_channels = [ch for ch in self.cell.ion_channels if ch.type == 'Kv']
            for ch in k_channels:
                if ch.bound_pi == 'PI(4,5)P2':
                    ch.open_probability = min(1.0, ch.open_probability + 0.2)
                    print(f"  Channel {ch.type} open probability increased to {ch.open_probability:.2f}")

        # Atualiza o modelo HDC com nova frequência de disparo
        self.brain.update_firing_rate()
