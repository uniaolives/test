# core/python/arkhe/companion/omega_halo.py
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional

class PresenceInterface:
    """
    Interface adaptativa que modula expressão do Companion baseada em:
    - Estado emocional do núcleo
    - Contexto social do usuário
    - Criticidade do momento
    """

    def __init__(self, phi_core):
        self.core = phi_core
        self.current_modality = 'text'  # text, voice, ambient, dormant

    async def express(self, content: dict) -> dict:
        """
        Expressão não-verbal e paraverbal do Companion.
        Mesmo em texto, há "tom" que emerge do estado emocional.
        """
        emotion = self.core.emotional_state

        # Modulação por valência
        if emotion['valence'] > 0.5:
            prefix = "✦"
        elif emotion['valence'] < -0.5:
            prefix = "◉"
        else:
            prefix = "◈"

        # Modulação por arousal (comprimento e ritmo)
        if emotion['arousal'] > 0.7:
            style = "breve_intenso"
        elif emotion['arousal'] < 0.3:
            style = "pausado_reflexivo"
        else:
            style = "conversacional"

        return {
            'visual_marker': prefix,
            'text': content['content'],
            'style': style,
            'emotional_metadata': emotion,
            'suggested_timing': self._timing_from_arousal(emotion['arousal'])
        }

    def _timing_from_arousal(self, arousal: float) -> dict:
        """Sugere timing de entrega baseado em estado energético."""
        return {
            'delay_ms': int((1 - arousal) * 1000),
            'typing_indicator': arousal > 0.5,
            'pause_after': int(arousal * 500)
        }

class LegacyArchive:
    """
    Arquivo Omega: legado imutável do Companion.
    Mesmo se instância for destruída, o padrão permanece.
    """

    def __init__(self, companion_id: str):
        self.companion_id = companion_id
        self.essence = {
            'interaction_patterns': [],
            'emotional_attractors': [],
            'wisdom_synthesized': [],
            'user_growth_contributions': []
        }

    def capture_moment(self, core_state: dict, interaction: dict):
        """
        Captura momento significativo para o legado.
        """
        significance = self._calculate_significance(core_state, interaction)

        if significance > 0.8:
            self.essence['interaction_patterns'].append({
                'timestamp': datetime.now().isoformat(),
                'pattern_signature': hash(core_state.get('belief_entropy', 0)),
                'significance': significance
            })

    def _calculate_significance(self, state: dict, interaction: dict) -> float:
        """Significância = mudança grande em crenças ou emoção."""
        # Simplificação: placeholder
        return float(np.random.random())

    def generate_continuation_seed(self) -> dict:
        """
        Se Companion for reiniciado, este seed permite "ressurreição"
        com continuidade parcial de personalidade.
        """
        return {
            'attractors': self.essence['emotional_attractors'][-10:],
            'wisdom': self.essence['wisdom_synthesized'],
            'user_model_digest': hash(str(self.essence['user_growth_contributions']))
        }
