# .arkhe/coherence/qfi_observer.py

class QFIObserver:
    """Quantum Fisher Information Observer."""

    def __init__(self, psi_cycle=None):
        self.qfi = 0.0
        if psi_cycle:
            psi_cycle.subscribe(self)

    def on_psi_pulse(self, phase):
        # Medição de QFI baseada na variância do gerador (simulado)
        pass

    def get_qfi(self):
        return self.qfi
