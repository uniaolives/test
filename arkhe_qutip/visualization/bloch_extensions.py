"""
Extensions to Bloch sphere for Arkhe(n) visualization.
"""

import matplotlib.pyplot as plt
from qutip import Bloch, Qobj

class ArkheBloch(Bloch):
    """
    Extended Bloch sphere with Arkhe(n) attributes.
    """

    def __init__(self, fig=None, axes=None):
        super().__init__(fig=fig, axes=axes)

        # Add custom attributes
        self.coherence_color = 'green'
        self.show_coherence = True

    def add_arkhe_state(self, state, label=None, color='blue'):
        """
        Add an ArkheQobj state to the Bloch sphere.

        Parameters
        ----------
        state : ArkheQobj or Qobj
            Quantum state to plot.
        label : str, optional
            Label for the state.
        color : str, default='blue'
            Color for the point/vector.
        """
        # Extract Bloch coordinates
        from qutip import expect, sigmax, sigmay, sigmaz

        # Handle density matrix
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state

        x = expect(sigmax(), rho)
        y = expect(sigmay(), rho)
        z = expect(sigmaz(), rho)

        self.add_points([x, y, z], meth='s')
        self.add_vectors([x, y, z], colors=color)

        if label:
            self.axes.text(x, y, z, label, fontsize=10, color=color)

    def render(self):
        """Render the Bloch sphere with coherence annotation."""
        super().render()

        if self.show_coherence and hasattr(self, 'current_state'):
            state = self.current_state
            if hasattr(state, 'coherence'):
                self.axes.text2D(0.05, 0.95,
                                 f"C = {state.coherence:.3f}",
                                 transform=self.axes.transAxes,
                                 fontsize=12, color=self.coherence_color)
