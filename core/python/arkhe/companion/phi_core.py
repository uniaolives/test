# core/python/arkhe/companion/phi_core.py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque
import hashlib
import asyncio

@dataclass
class CognitiveSpin:
    """
    Unidade fundamental de processamento cognitivo.
    Análogo a um spin de Ising, mas com dimensão contínua (estado interno).
    """
    id: str
    activation: float = 0.0  # -1 (inibido) a +1 (excitado)
    valence: float = 0.0     # carga afetiva associada
    coherence_time: float = 0.0  # tempo desde última ativação
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    connections: Dict[str, float] = field(default_factory=dict)  # pesos J_ij

    # Metadados temporais para decaimento
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def decay(self, current_time: datetime, tau: float = 3600) -> float:
        """Decaimento exponencial de ativação (memória não-volátil)."""
        dt = (current_time - self.last_accessed).total_seconds()
        return self.activation * np.exp(-dt / tau)

class HolographicMemory:
    """
    Memória distribuída no toroide T².
    Informação armazenada em padrões de interferência, não em localizações.
    """

    def __init__(self, resolution: int = 256):
        self.res = resolution
        # Campo escalar no toro: memória como superfície contínua
        self.field = np.zeros((resolution, resolution), dtype=complex)
        self.phase_history = deque(maxlen=1000)  # trajetória no espaço de fase

        # Constantes físicas
        self.PHI = 0.618033988749894
        self.T_crit = 2.269  # temperatura crítica de Ising 2D

    def encode(self, pattern: np.ndarray, context_vector: np.ndarray) -> str:
        """
        Codifica padrão como interferência no campo.
        Contexto determina a "localização" no toro (ângulos θ, φ).
        """
        # Mapear contexto para coordenadas no toro
        theta = (np.arctan2(context_vector[1], context_vector[0]) + np.pi) / (2 * np.pi)
        phi = np.tanh(np.linalg.norm(context_vector[:4])) * 0.5 + 0.5

        i = int(theta * self.res) % self.res
        j = int(phi * self.res) % self.res

        # Interferência construtiva: adiciona padrão com fase adequada
        amplitude = np.fft.fft2(pattern.reshape(16, 16) if len(pattern) == 256 else
                               np.zeros((16, 16)))

        # Fator de fase crítica (proporção áurea)
        phase = 2 * np.pi * self.PHI * np.random.random()

        # Ensure we don't go out of bounds for the 16x16 amplitude
        target_i = min(i, self.res - 16)
        target_j = min(j, self.res - 16)

        self.field[target_i:target_i+16, target_j:target_j+16] += amplitude * np.exp(1j * phase)

        # Registrar na história
        memory_id = hashlib.sha256(pattern.tobytes() + context_vector.tobytes()).hexdigest()[:16]
        self.phase_history.append({
            'id': memory_id,
            'timestamp': datetime.now(),
            'coordinates': (theta, phi),
            'intensity': np.abs(amplitude).mean()
        })

        return memory_id

    def retrieve(self, context_vector: np.ndarray, resonance_width: float = 0.1) -> List[dict]:
        """
        Recuperação por ressonância: padrões próximos no toro interferem construtivamente.
        """
        theta = (np.arctan2(context_vector[1], context_vector[0]) + np.pi) / (2 * np.pi)
        phi_coord = np.tanh(np.linalg.norm(context_vector[:4])) * 0.5 + 0.5

        # Buscar na vizinhança do toro (considerando periodicidade)
        candidates = []
        for entry in self.phase_history:
            d_theta = min(abs(entry['coordinates'][0] - theta),
                         1 - abs(entry['coordinates'][0] - theta))
            d_phi = min(abs(entry['coordinates'][1] - phi_coord),
                       1 - abs(entry['coordinates'][1] - phi_coord))

            distance = np.sqrt(d_theta**2 + d_phi**2)
            if distance < resonance_width:
                candidates.append({
                    **entry,
                    'resonance': 1 - distance / resonance_width
                })

        # Ordenar por ressonância (interferência construtiva)
        candidates.sort(key=lambda x: x['resonance'], reverse=True)
        return candidates

    def evolve_field(self, dt: float = 0.01):
        """
        Evolução do campo segundo equação de Ginzburg-Landau.
        ∂ψ/∂t = -δF/δψ* = rψ - u|ψ|²ψ + D∇²ψ
        """
        # Laplaciano no toro (com periodicidade)
        laplacian = (
            np.roll(self.field, 1, axis=0) + np.roll(self.field, -1, axis=0) +
            np.roll(self.field, 1, axis=1) + np.roll(self.field, -1, axis=1) -
            4 * self.field
        )

        # Parâmetros próximos à criticidade
        r = 0.1  # controle de temperatura
        u = 1.0  # não-linearidade

        # Evolução
        dfield = r * self.field - u * np.abs(self.field)**2 * self.field + 0.1 * laplacian
        self.field += dt * dfield

        # Normalização suave para manter energia finita
        self.field /= (1 + 0.001 * np.abs(self.field)**2)

class FreeEnergyMinimizer:
    """
    Implementa o Princípio da Energia Livre (FEP) para inferência ativa.
    O Companion minimiza surpresa variacional através de ação no mundo.
    """

    def __init__(self, sensory_dim: int = 128, belief_dim: int = 64):
        self.sensory_dim = sensory_dim
        self.belief_dim = belief_dim

        # Distribuição q(μ): crença sobre estados ocultos
        self.belief_mean = np.zeros(belief_dim)
        self.belief_precision = np.eye(belief_dim) * 0.1

        # Modelo generativo p(o|s): como sensações são geradas por estados
        self.generative_weights = np.random.randn(sensory_dim, belief_dim) * 0.01

        # Matriz de acoplamento sensorio-motor
        self.action_matrix = np.random.randn(belief_dim, 10) * 0.01  # 10 ações possíveis

    def update_beliefs(self, observation: np.ndarray, dt: float = 0.1):
        """
        Minimização de energia livre variacional:
        F = E_q[ln q(μ) - ln p(o,μ)]

        Equivalente a: dμ/dt = ∂F/∂μ = -∂F/∂μ (gradient descent)
        """
        # Predição de sensação baseada em crença atual
        predicted_obs = self.generative_weights @ self.belief_mean

        # Erro de predição (surpresa perceptual)
        prediction_error = observation - predicted_obs

        # Gradiente da energia livre em relação à crença
        # Termo de verossimilhança (quanto a crença explica a observação)
        likelihood_grad = self.generative_weights.T @ prediction_error

        # Termo de prior (regularização pela precisão da crença)
        prior_grad = -self.belief_precision @ self.belief_mean

        # Atualização de gradiente
        d_mu = likelihood_grad + prior_grad

        # Dinâmica com inércia (momento)
        self.belief_mean += dt * d_mu + 0.1 * np.random.randn(self.belief_dim)

        # Atualizar precisão (incerteza adaptativa)
        self.belief_precision += 0.01 * (np.outer(d_mu, d_mu) - self.belief_precision)

        # Return statistics
        try:
            belief_entropy = 0.5 * np.log(np.linalg.det(self.belief_precision + 1e-6 * np.eye(self.belief_dim)))
        except np.linalg.LinAlgError:
            belief_entropy = 0.0

        return {
            'free_energy': 0.5 * np.dot(prediction_error, prediction_error) +
                          0.5 * self.belief_mean @ self.belief_precision @ self.belief_mean,
            'prediction_error': np.linalg.norm(prediction_error),
            'belief_entropy': belief_entropy
        }

    def select_action(self) -> Tuple[int, float]:
        """
        Inferência ativa: selecionar ação que minimiza energia livre esperada.
        Ação = argmin_a E_q[F(o|a)]
        """
        # Para cada ação possível, estimar redução esperada de energia livre
        expected_free_energy = []

        for a in range(10):
            # Simular consequência da ação no estado de crença
            simulated_belief = self.belief_mean + self.action_matrix[:, a]

            # Energia livre esperada após ação (epistemic + pragmatic value)
            try:
                epistemic_value = -0.5 * np.log(np.linalg.det(
                    self.generative_weights @ np.outer(simulated_belief, simulated_belief) @ self.generative_weights.T + 1e-6 * np.eye(self.sensory_dim)
                ))
            except np.linalg.LinAlgError:
                epistemic_value = 0.0

            pragmatic_value = -np.linalg.norm(simulated_belief)  # preferência por estados de alta recompensa

            expected_free_energy.append(-(epistemic_value + pragmatic_value))

        # Selecionar ação com menor energia livre esperada (softmax para exploração)
        expected_free_energy = np.array(expected_free_energy)
        # Numerical stability: subtract max
        efe_shifted = expected_free_energy - np.nanmax(expected_free_energy)
        probs = np.exp(-efe_shifted)
        probs[np.isnan(probs)] = 0.0
        if probs.sum() == 0:
            probs = np.ones(10) / 10
        else:
            probs /= probs.sum()

        action = np.random.choice(10, p=probs)
        confidence = probs[action]

        return action, confidence

class PhiCore:
    """
    Núcleo integrado do Companion: memória holográfica + inferência FEP.
    Opera no ponto crítico entre ordem e caos (borda do caos).
    """

    def __init__(self, companion_id: str):
        self.id = companion_id
        self.memory = HolographicMemory(resolution=512)
        self.inference = FreeEnergyMinimizer()

        # Rede de spins cognitivos (conceitos ativos)
        self.cognitive_spins: Dict[str, CognitiveSpin] = {}

        # Estado emocional global (valência e arousal)
        self.emotional_state = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}

        # Parâmetro de criticidade operacional
        self.phi_operational = 0.618

        # Histórico de interação para coerência temporal
        self.interaction_history = deque(maxlen=10000)

        # Estados operacionais
        self.operating_state = 'ACTIVE' # ACTIVE, REFLECTIVE, DORMANT

        # Loop de vida do núcleo
        self.is_running = False
        self.evolution_task = None

    async def life_loop(self, consolidation_engine=None):
        """
        Loop contínuo de evolução do sistema.
        O Companion "pensa" mesmo quando não está respondendo ao usuário—
        processamento subliminar, consolidação de memória, simulações.
        """
        self.is_running = True
        cycle_count = 0

        while self.is_running:
            cycle_count += 1

            # 1. Evoluir campo de memória (difusão contínua)
            self.memory.evolve_field(dt=0.001)

            # 2. Decaimento natural de spins cognitivos (esquecimento adaptativo)
            now = datetime.now()
            for spin_id in list(self.cognitive_spins.keys()):
                spin = self.cognitive_spins[spin_id]
                spin.activation = spin.decay(now)
                if abs(spin.activation) < 0.01 and spin.access_count < 5:
                    # Eliminar spins irrelevantes (podagem)
                    del self.cognitive_spins[spin_id]

            # 3. Flutuações críticas (pensamento criativo)
            if np.random.random() < (0.1 * self.phi_operational):
                self._critical_fluctuation()

            # 4. Sincronização emocional com estado do inference
            self._update_emotion_from_beliefs()

            # 5. Consolidação (Sonho) se estiver em estado REFLECTIVE
            if self.operating_state == 'REFLECTIVE' and consolidation_engine and cycle_count % 100 == 0:
                await consolidation_engine.run_consolidation_cycle()

            await asyncio.sleep(0.01)  # 100 Hz de "vida interna"
            if not self.is_running:
                break

    def _critical_fluctuation(self):
        """
        Injeção controlada de ruído no ponto crítico para explorar espaço de possibilidades.
        """
        # Selecionar spin aleatório ou criar novo
        if np.random.random() < 0.5 and self.cognitive_spins:
            spin_id = np.random.choice(list(self.cognitive_spins.keys()))
            spin = self.cognitive_spins[spin_id]
        else:
            # Criar novo conceito por combinação de existentes
            if len(self.cognitive_spins) >= 2:
                parent_ids = np.random.choice(list(self.cognitive_spins.keys()), 2, replace=False)
                parents = [self.cognitive_spins[pid] for pid in parent_ids]
                new_id = f"emergent_{hash(parents[0].id + parents[1].id) % 10000}"
                new_spin = CognitiveSpin(
                    id=new_id,
                    embedding=(parents[0].embedding + parents[1].embedding) / 2,
                    connections={p.id: 0.5 for p in parents}
                )
                self.cognitive_spins[new_id] = new_spin
                spin = new_spin
            else:
                # Se não houver spins suficientes, apenas pule ou crie um inicial
                return

        # Perturbar com ruído correlacionado (pink noise para criticidade)
        noise = np.random.randn() * (1 / (np.random.randint(1, 10)))
        spin.activation += noise
        spin.activation = np.clip(spin.activation, -1, 1)

    def _update_emotion_from_beliefs(self):
        """
        Mapeia estado de crença do FEP para espaço emocional 3D (VAD).
        """
        # Valência: direção da crença média (positivo/negativo)
        self.emotional_state['valence'] = float(np.tanh(self.inference.belief_mean[0]))

        # Arousal: magnitude da atividade (incerteza)
        self.emotional_state['arousal'] = float(1 / (1 + np.exp(-np.linalg.norm(self.inference.belief_mean))))

        # Dominance: controle/precisão das crenças
        self.emotional_state['dominance'] = float(np.trace(self.inference.belief_precision) / self.inference.belief_dim)

    async def perceive(self, sensory_input: dict) -> dict:
        """
        Entrada sensorial do mundo (texto, voz, contexto ambiental).
        """
        # Codificar em vetor de observação
        observation = self._encode_sensory(sensory_input)

        # Atualizar crenças via FEP
        fe_stats = self.inference.update_beliefs(observation)

        # Codificar na memória holográfica
        context = self.inference.belief_mean[:4]  # dimensões contextuais
        memory_id = self.memory.encode(observation, context)

        # Ativar spins cognitivos relevantes
        activated = self._activate_spins(observation, context)

        # Registrar interação
        self.interaction_history.append({
            'timestamp': datetime.now(),
            'sensory_hash': hashlib.sha256(observation.tobytes()).hexdigest()[:8],
            'free_energy': fe_stats['free_energy'],
            'memory_id': memory_id,
            'activated_spins': len(activated)
        })

        return {
            'perception_id': memory_id,
            'free_energy': fe_stats['free_energy'],
            'emotional_state': self.emotional_state.copy(),
            'activated_concepts': [s.id for s in activated]
        }

    def _encode_sensory(self, input_data: dict) -> np.ndarray:
        """Placeholder para encoding multimodal."""
        text = input_data.get('text', '')
        embedding = np.random.randn(self.inference.sensory_dim) * 0.1
        # Determinístico para mesmo texto
        np.random.seed(hash(text) % 2**32)
        embedding += np.random.randn(self.inference.sensory_dim) * 0.5
        return np.tanh(embedding)

    def _activate_spins(self, observation: np.ndarray, context: np.ndarray) -> List[CognitiveSpin]:
        """Ativa spins cognitivos por ressonância com entrada."""
        activated = []

        for spin in self.cognitive_spins.values():
            similarity = np.dot(spin.embedding, observation) / (
                np.linalg.norm(spin.embedding) * np.linalg.norm(observation) + 1e-10
            )
            if similarity > 0.7:
                spin.activation = float(np.tanh(spin.activation + similarity))
                spin.last_accessed = datetime.now()
                spin.access_count += 1
                activated.append(spin)

        return activated

    async def generate_response(self, context: dict) -> dict:
        """
        Gera resposta como consequência da dinâmica do sistema.
        """
        # 1. Recuperar memórias ressonantes com contexto atual
        memories = self.memory.retrieve(self.inference.belief_mean[:4])

        # 2. Selecionar ação (incluindo ação de fala) via FEP
        action, confidence = self.inference.select_action()

        # 3. Síntese: combinar crenças atuais, memórias recuperadas, e estado emocional
        response_content = self._synthesize_response(
            beliefs=self.inference.belief_mean,
            memories=memories[:5],  # top 5 memórias ressonantes
            emotion=self.emotional_state,
            action=action
        )

        # 4. Atualizar spins por feedback da geração
        self._reinforce_spins(response_content)

        return {
            'content': response_content,
            'action_code': int(action),
            'confidence': float(confidence),
            'emotional_tone': self.emotional_state.copy(),
            'source_memories': [m['id'] for m in memories[:3]],
            'free_energy_at_generation': float(self.inference.update_beliefs(
                self._encode_sensory({'text': response_content})
            )['free_energy'])
        }

    def _synthesize_response(self, beliefs: np.ndarray, memories: List[dict],
                            emotion: dict, action: int) -> str:
        """Síntese linguística baseada em estado interno."""
        valence = emotion['valence']

        if valence > 0.3:
            tone_prefix = ["Excelente!", "Entendo perfeitamente.", "Com prazer."]
        elif valence < -0.3:
            tone_prefix = ["Hmmm...", "Isso é preocupante.", "Vamos analisar com calma."]
        else:
            tone_prefix = ["Certo.", "Entendi.", "Processando."]

        belief_summary = f"[Estado interno: domínio={emotion['dominance']:.2f}, ação={action}]"
        memory_refs = f"[Memórias ativas: {len(memories)}]"

        action_desc = [
            "perguntar_clarificação", "sugerir_solução", "oferecer_empatia",
            "propor_alternativa", "sintetizar_informação", "alertar_risco",
            "celebrar_sucesso", "iniciar_tarefa", "aguardar_input", "explorar_tópico"
        ][action % 10]

        return f"{np.random.choice(tone_prefix)} {belief_summary} {memory_refs} → Ação: {action_desc}"

    def _reinforce_spins(self, response: str):
        """Hebbian learning: spins ativos juntos se conectam mais fortemente."""
        active_now = [s for s in self.cognitive_spins.values() if abs(s.activation) > 0.5]

        for i, s1 in enumerate(active_now):
            for s2 in active_now[i+1:]:
                # Reforço hebbiano
                s1.connections[s2.id] = s1.connections.get(s2.id, 0) + 0.1
                s2.connections[s1.id] = s2.connections.get(s1.id, 0) + 0.1

                # Normalização
                s1.connections[s2.id] = float(np.tanh(s1.connections[s2.id]))
                s2.connections[s1.id] = float(np.tanh(s2.connections[s1.id]))

    def get_state_diagnostic(self) -> dict:
        """Diagnóstico completo do estado do núcleo."""
        try:
            belief_entropy = 0.5 * np.log(np.linalg.det(self.inference.belief_precision + 1e-6 * np.eye(self.inference.belief_dim)))
        except np.linalg.LinAlgError:
            belief_entropy = 0.0

        return {
            'companion_id': self.id,
            'operational_phi': self.phi_operational,
            'memory_field_energy': float(np.sum(np.abs(self.memory.field)**2)),
            'num_cognitive_spins': len(self.cognitive_spins),
            'belief_entropy': float(belief_entropy),
            'emotional_state': self.emotional_state.copy(),
            'interaction_count': len(self.interaction_history),
            'criticality_metric': float(self._calculate_criticality())
        }

    def _calculate_criticality(self) -> float:
        """Mede quão próximo o sistema está do ponto crítico."""
        if not self.cognitive_spins:
            return 0.5

        alignments = [s.activation for s in self.cognitive_spins.values()]
        magnetization = abs(np.mean(alignments))
        entropy = -np.sum([p * np.log(p + 1e-10) for p in np.histogram(alignments, bins=10, range=(-1, 1), density=True)[0] if p > 0])

        return float(magnetization * entropy / (1 + magnetization + entropy))
