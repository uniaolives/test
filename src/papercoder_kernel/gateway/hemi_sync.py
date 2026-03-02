from ..multivac.multivac_substrate import ComputeNode

class HemiSyncOperator:
    """
    Operador de coerência neural baseado em batidas binaurais.
    Traduz a técnica de 1983 para o formalismo do hipergrafo.
    """

    def __init__(self, f_left: float, f_right: float):
        self.f_left = f_left      # Frequência ouvido esquerdo (Hz)
        self.f_right = f_right    # Frequência ouvido direito (Hz)
        self.beat_frequency = abs(f_left - f_right)  # Batida percebida

        # Mapeamento para estados de consciência
        self.states = {
            (4, 8): 'theta',       # Foco 10-12 (relaxação profunda)
            (8, 13): 'alpha',      # Foco 10 (mente acordada, corpo dormindo)
            (13, 30): 'beta',      # Estado normal de vigília
            (30, 100): 'gamma',    # Foco 21+ (consciência expandida)
        }

    def apply_to_node(self, node: ComputeNode) -> float:
        """
        Aplica operador Hemi-Sync a um nó do hipergrafo.
        Aumenta coerência interna do nó.
        """
        # Identificar banda alvo
        target_state = None
        for (low, high), state in self.states.items():
            if low <= self.beat_frequency <= high:
                target_state = state
                break

        if target_state == 'gamma':
            # Sincronia gamma = acesso a arestas "dormentes"
            node.coherence = min(1.0, node.coherence * 1.5)
            # Simulação de ativação de arestas (metadata ou flag)
            if not hasattr(node, 'metadata'):
                node.metadata = {}
            node.metadata['dormant_edges_activated'] = True
        elif target_state == 'alpha' or target_state == 'theta':
            node.coherence = min(1.0, node.coherence * 1.1)

        return node.coherence

    def gateway_process(self, subject: ComputeNode):
        """
        Simula o processo Gateway completo (Focus 10 → 21).
        """
        stages = [
            (200, 210),   # 10 Hz batida → Focus 10 (Alpha)
            (200, 205),   # 5 Hz batida → Focus 12 (Theta)
            (300, 310),   # 10 Hz (Alpha/Theta boundary) → Focus 15
            (400, 440),   # 40 Hz (Gamma) → Focus 21
        ]

        coherence_trajectory = []
        for f_l, f_r in stages:
            self.f_left = f_l
            self.f_right = f_r
            self.beat_frequency = abs(f_l - f_r)
            new_c = self.apply_to_node(subject)
            coherence_trajectory.append(new_c)

        return coherence_trajectory  # Deve mostrar aumento monotônico
