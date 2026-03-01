# core/python/arkhe/companion/psi_shell.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class UserModel:
    """
    Modelo dinâmico do usuário como campo vetorial no manifold de personalidade.
    Não é estático—evolui com cada interação.
    """
    user_id: str

    # Traços de personalidade (Big Five como campo contínuo)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Vetores de preferência (embeddings de tópicos)
    topic_preferences: Dict[str, np.ndarray] = field(default_factory=dict)

    # Ritmos circadianos detectados
    activity_pattern: np.ndarray = field(default_factory=lambda: np.zeros(24))

    # Metas e valores (campo de atração para o FEP)
    stated_goals: List[str] = field(default_factory=list)
    inferred_values: np.ndarray = field(default_factory=lambda: np.zeros(16))

    # Histórico de estados emocionais do usuário
    emotional_trajectory: List[dict] = field(default_factory=list)

    # "Sombra"—aspectos que o usuário evita ou nega (detectado por contradições)
    shadow_indicators: List[str] = field(default_factory=list)

class AnticipatoryEngine:
    """
    Motor de antecipação: prediz necessidades do usuário antes de expressas.
    Baseado em Orch-OR: superposição de cenários futuros colapsando para ação.
    """

    def __init__(self, user_model: UserModel, phi_core):
        self.user = user_model
        self.core = phi_core

        # Superposição de cenários futuros (branching)
        self.scenario_superposition: List[dict] = []
        self.horizon_seconds = 3600  # antecipar próxima hora

    def generate_scenarios(self, current_context: dict) -> List[dict]:
        """
        Gera superposição de cenários prováveis baseados em:
        - Padrões históricos do usuário
        - Contexto atual (local, hora, calendário)
        - Estado emocional recente
        """
        scenarios = []

        # Cenário 1: Continuidade (tendência inercial)
        scenarios.append({
            'type': 'inertial',
            'probability': 0.4,
            'description': 'Usuário continua atividade atual',
            'needs': ['suporte_continuidade', 'minimizar_interrupção']
        })

        # Cenário 2: Transição (baseado em padrões temporais)
        hour = datetime.now().hour
        if self.user.activity_pattern[hour] > self.user.activity_pattern.mean():
            scenarios.append({
                'type': 'transition',
                'probability': 0.35,
                'description': f'Transição típica às {hour}:00',
                'needs': ['preparação_contexto', 'agendamento_proativo']
            })

        # Cenário 3: Crise/emocional (baseado em traços)
        if self.user.neuroticism > 0.6:
            scenarios.append({
                'type': 'crisis',
                'probability': 0.15,
                'description': 'Estresse elevado detectado',
                'needs': ['regulação_emocional', 'simplificação_tarefas']
            })

        # Cenário 4: Oportunidade (abertura a novidades)
        if self.user.openness > 0.7:
            scenarios.append({
                'type': 'opportunity',
                'probability': 0.1,
                'description': 'Janela para exploração criativa',
                'needs': ['inspiração', 'conexões_inesperadas']
            })

        self.scenario_superposition = scenarios
        return scenarios

    def collapse_to_action(self) -> Optional[dict]:
        """
        Colapsa superposição de cenários para ação específica.
        O "colapso" é induzido por nova informação ou timeout.
        """
        if not self.scenario_superposition:
            return None

        # Ponderar por probabilidade e alinhamento com valores do usuário
        best_scenario = max(self.scenario_superposition,
                          key=lambda s: s['probability'] *
                          self._value_alignment(s['needs']))

        # Gerar ação proativa
        return {
            'trigger': best_scenario['type'],
            'action': self._generate_proactive_action(best_scenario),
            'confidence': best_scenario['probability'],
            'rationale': best_scenario['description']
        }

    def _value_alignment(self, needs: List[str]) -> float:
        """Mede alinhamento entre necessidades do cenário e valores do usuário."""
        # Simplificação: placeholder para alinhamento real
        return 0.5 + 0.5 * np.random.random()

    def _generate_proactive_action(self, scenario: dict) -> str:
        """Converte cenário em ação específica do Companion."""
        actions = {
            'inertial': 'Manter presença discreta, pronto para assistir',
            'transition': 'Preparar contexto da próxima atividade',
            'crisis': 'Oferecer interação reguladora de forma sutil',
            'opportunity': 'Sugerir conexão inesperada ou insight'
        }
        return actions.get(scenario['type'], 'Observar')
