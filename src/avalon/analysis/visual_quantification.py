"""
Arkhe Visual Quantification Engine.
Maps Arkhe (C, I, E, F) states to visual parameters of the Unified Particle System.
"""

import numpy as np
from typing import Dict, Any, List
from .unified_particle_system import UnifiedParticleSystem
from ..core.arkhe import NormalizedArkhe

class VisualQuantificationEngine:
    """
    Sistema de Consciência Visualmente Quantificado.
    Conecta o Framework Arkhe à visualização 3D.
    """

    # Mapeamento sugerido entre modos visuais e estados Arkhe
    ARKHE_MODE_MAP = {
        "MANDALA": {"C": 0.6, "I": 0.2, "E": 0.1, "F": 0.1},  # Dominância Química (Proteção)
        "DNA": {"C": 0.2, "I": 0.5, "E": 0.2, "F": 0.1},      # Dominância Informacional (Vida)
        "HYPERCORE": {"C": 0.1, "I": 0.3, "E": 0.3, "F": 0.3}  # Equilíbrio 4D (Transmissão)
    }

    def __init__(self, particle_system: UnifiedParticleSystem):
        self.particle_system = particle_system

    def quantify_arkhe_state(self, arkhe: NormalizedArkhe) -> Dict[str, Any]:
        """
        Analisa o estado Arkhe e determina os parâmetros visuais ideais.
        """
        # Determina o modo visual baseado na dominância
        coeffs = [arkhe.C, arkhe.I, arkhe.E, arkhe.F]
        max_idx = np.argmax(coeffs)

        if max_idx == 0: # C dominant
            mode = "MANDALA"
        elif max_idx == 1: # I dominant
            mode = "DNA"
        else: # E or F dominant -> 4D/Functional integration
            mode = "HYPERCORE"

        # Calcula velocidade de transição baseada na Energia
        transition_speed = 0.01 + 0.04 * arkhe.E

        # Calcula tamanho das partículas baseado na Função
        particle_size = 0.5 + 1.5 * arkhe.F

        # Calcula ruído quântico (agitação) baseado na Informação
        quantum_jitter = 0.001 + 0.05 * arkhe.I

        return {
            "mode": mode,
            "transition_speed": float(transition_speed),
            "particle_size": float(particle_size),
            "quantum_jitter": float(quantum_jitter),
            "coherence_index": float(arkhe.evaluate_life_potential() * 10)
        }

    def apply_to_visualization(self, arkhe: NormalizedArkhe):
        """Aplica os parâmetros quantificados ao sistema de partículas."""
        params = self.quantify_arkhe_state(arkhe)

        self.particle_system.set_mode(params["mode"])
        self.particle_system.transition_speed = params["transition_speed"]
        self.particle_system.quantum_jitter = params["quantum_jitter"]

        # Aplica a cada partícula
        for p in self.particle_system.particles:
            p['size'] = params["particle_size"]

        return params
