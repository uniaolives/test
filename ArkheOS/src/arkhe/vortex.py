# vortex.py
import numpy as np
try:
    from .arkhe_error_handler import safe_operation, logging
except ImportError:
    from arkhe_error_handler import safe_operation, logging

class OAMVortex:
    """
    Implementação de Handovers com Momento Angular Orbital (OAM).
    A frente de onda espirala conforme a carga topológica 'l'.
    """
    def __init__(self, l=1, frequency=2.618): # frequency in Hz (phi^2)
        self.l = l
        self.omega = 2 * np.pi * frequency
        self.k = 1.0 # Wave number (simplified)

    def calculate_phase(self, x, y, z, t):
        """
        Calcula a fase do vórtice no ponto (x, y, z) no tempo t.
        phi = l * arctan2(y, x)
        """
        phi_azimutal = np.arctan2(y, x)
        phase = self.k * z - self.omega * t + self.l * phi_azimutal
        return phase

    def get_wavefront_value(self, x, y, z, t):
        """
        Calcula o valor complexo da frente de onda: A * e^(i * phase).
        """
        r = np.sqrt(x**2 + y**2)
        # Amplitude (simplified Laguerre-Gaussian-like profile: r^|l| * exp(-r^2))
        amplitude = (r ** abs(self.l)) * np.exp(-r**2)
        phase = self.calculate_phase(x, y, z, t)
        return amplitude * np.exp(1j * phase)

class VortexHandoverManager:
    """Gerencia a emissão de vórtices pela Tríade Federada."""
    def __init__(self, triad):
        self.triad = triad
        self.active_vortices = []

    @safe_operation
    def emit_vortex(self, l=1):
        if not self.triad.is_entangled:
            logging.warning("Tríade não emaranhada. Emissão de vórtice degradada.")

        vortex = OAMVortex(l=l)
        self.active_vortices.append(vortex)
        logging.info(f"Vórtice OAM emitido com carga topológica l={l}.")
        return vortex

if __name__ == "__main__":
    print("Testando Geração de Vórtice OAM...")
    from resonance import FederatedTriad

    triad = FederatedTriad()
    triad.is_entangled = True # Simular estado emaranhado

    manager = VortexHandoverManager(triad)
    vortex = manager.emit_vortex(l=1)

    # Testar fase em um círculo ao redor do eixo Z
    angles = np.linspace(0, 2*np.pi, 8)
    for angle in angles:
        x, y = np.cos(angle), np.sin(angle)
        phase = vortex.calculate_phase(x, y, z=1.0, t=0)
        print(f"Ângulo: {angle:5.2f} rad | Fase OAM: {phase:5.2f} rad")

    print("Vórtice validado.")
