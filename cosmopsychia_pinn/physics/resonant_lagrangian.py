"""
resonant_lagrangian.py
Implementação do modelo físico de ressonância Chronoflux
Baseado no formalismo de Lagrangianas acopladas
"""

import numpy as np

class ResonantChronofluxModel:
    """
    Modelo de Chronoflux Resonante.
    Mapeia a interação entre o campo de fundo e o detector sintonizado.
    """
    def __init__(self, Q: float = 100.0, coupling_factor: float = 0.5):
        self.Q = Q  # Fator de qualidade do ressonador
        self.coupling = coupling_factor

    def calculate_resonance_amplification(self, freq: float, target_freq: float = 0.00783) -> float:
        """
        Calcula a amplificação via curva de ressonância Lorentziana.
        target_freq padrão: 0.00783 Hz (Frequência de Schumann / 1000? ou sintonização específica)
        """
        # Evita divisão por zero
        if freq == 0: return 0.0

        # Resposta de amplitude Lorentziana
        # A(f) = (f * f0 / Q) / sqrt((f0^2 - f^2)^2 + (f * f0 / Q)^2)
        # Simplificado para ganho relativo
        numerator = (freq * target_freq / self.Q)**2
        denominator = (target_freq**2 - freq**2)**2 + (freq * target_freq / self.Q)**2
        return np.sqrt(numerator / denominator)

    def lagrangian_density(self, psi_dot, phi_dot, psi, phi, coupling_g=0.1):
        """
        Densidade de Lagrangiana para o sistema acoplado.
        L = 1/2 (psi_dot^2 - m^2 psi^2) + 1/2 (phi_dot^2 - omega^2 phi^2) + g * psi * phi
        """
        m = 1.0
        omega = 1.0
        kinetic = 0.5 * (psi_dot**2 + phi_dot**2)
        potential = 0.5 * (m**2 * psi**2 + omega**2 * phi**2)
        interaction = coupling_g * psi * phi
        return kinetic - potential + interaction

def simulate_chronoflux_interaction(duration=1000, Q=500, noise_floor=1.0):
    """
    Simula a interação do sinal Chronoflux com um detector ressonante.
    """
    model = ResonantChronofluxModel(Q=Q)
    time = np.linspace(0, duration, duration)

    # Ruído cósmico / de vácuo
    noise = np.random.normal(0, noise_floor, duration)

    # Frequência de sintonização (ex: Schumann Resonance)
    f0 = 0.00783

    # Amplificação no pico de ressonância
    gain = model.calculate_resonance_amplification(f0, target_freq=f0)

    # Sinal amplificado pela ressonância (coupling * gain)
    signal_strength = 0.1 * model.coupling * gain
    signal = signal_strength * np.sin(2 * np.pi * f0 * time)

    observation = noise + signal

    # Cálculo de SNR em dB
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    snr_db = 10 * np.log10(signal_power / noise_power) if signal_power > 0 else -float('inf')

    return time, observation, snr_db
