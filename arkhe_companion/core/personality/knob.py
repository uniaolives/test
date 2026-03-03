# core/python/arkhe/companion/personality.py
import numpy as np
from datetime import datetime
from typing import Optional, Dict

class NonLinearPersonalityKnob:
    """
    Interface de temperatura φ com barreiras de zona e transições suaves.
    """
    ZONES = {
        'analytical': (0.0, 0.25),
        'balanced': (0.25, 0.75),
        'creative': (0.75, 1.0)
    }

    def __init__(self, core):
        self.core = core
        self.phi = 0.618
        self.zone_inertia = 0.1

    def _get_zone(self, val: float) -> str:
        for zone, (low, high) in self.ZONES.items():
            if low <= val <= high:
                return zone
        return 'balanced'

    def set_phi(self, target: float):
        target = np.clip(target, 0.0, 1.0)
        old_zone = self._get_zone(self.phi)
        new_zone = self._get_zone(target)

        # Inércia de zona
        if old_zone != new_zone:
            if abs(target - self.phi) < self.zone_inertia:
                return

        # Interpolação suave
        self.phi = self.phi * 0.7 + target * 0.3
        self._apply_phi()

    def _apply_phi(self):
        """Propaga φ para subsistemas do core."""
        # 1. Temperatura do FEP (mapeia para [0, 2])
        self.core.inference.temperature = self.phi * 2.0

        # 2. Intensidade de flutuações críticas
        self.core.phi_operational = self.phi

class ContextualPhi:
    """
    Ajusta φ automaticamente baseado no contexto detectado (horário, stress).
    """
    def __init__(self, knob: NonLinearPersonalityKnob):
        self.knob = knob
        self.profiles = {
            'work': 0.2,
            'leisure': 0.8,
            'social': 0.6,
            'crisis': 0.1
        }
        self.manual_override = None

    def update_context(self, signals: Dict):
        """ML leve / Heurística para contexto."""
        hour = datetime.now().hour
        context = 'leisure'

        if 9 <= hour <= 18:
            context = 'work'

        if signals.get('stress_level', 0) > 0.7:
            context = 'crisis'
        elif signals.get('social_activity'):
            context = 'social'

        target_phi = self.profiles.get(context, 0.5)
        self.knob.set_phi(target_phi)
        return context
