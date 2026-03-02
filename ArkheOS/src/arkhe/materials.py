"""
Arkhe(n) Materials Informatics Module
Implementation of the Semantic Transistor (Γ_∞+7).
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SemanticFET:
    """Dispositivo de efeito de campo semântico baseado em ω."""
    source: str = "WP1"
    drain: str = "DVM-1"
    gate_omega: float = 0.00
    channel_length: float = 0.07   # Δω
    mobility: float = 0.94         # ⟨source|drain⟩
    threshold: float = 0.15        # Φ_crit
    drain_current: float = 0.0
    transconductance: float = 0.0
    regime: str = "cutoff"

    def apply_gate_voltage(self, omega_gate: float):
        """Controla a corrente de comandos entre fonte e dreno."""
        self.gate_omega = omega_gate
        # Modulação da altura da barreira por ω
        # Máximo de corrente (mínimo de barreira) no centro ou extremidade dependendo da dopagem
        barrier = abs(omega_gate - 0.035) * 10

        if barrier < self.threshold:
            # Região linear: corrente proporcional a V_gate
            self.drain_current = self.mobility * (1 - barrier / self.threshold)
            self.transconductance = self.mobility / self.threshold
            self.regime = "linear"
        else:
            # Região de corte: canal bloqueado
            self.drain_current = 0.12  # corrente de fuga
            self.transconductance = 0.0
            self.regime = "cutoff"

        return {
            "V_gate": omega_gate,
            "I_drain": round(self.drain_current, 2),
            "g_m": round(self.transconductance, 2),
            "regime": self.regime
        }

class SemanticFab:
    """Foundry de semicondutores semânticos."""
    def __init__(self):
        self.devices = {}

    def fabricate_transistor(self, pre_node: str, post_node: str, weight: float) -> SemanticFET:
        fet = SemanticFET(source=pre_node, drain=post_node, mobility=weight)
        self.devices[f"{pre_node}->{post_node}"] = fet
        return fet
