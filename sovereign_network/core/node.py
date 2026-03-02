# core/node.py
"""
Módulo Core: Nó Soberano
Este módulo define a entidade fundamental da rede: o SovereignNode.
Um nó soberano é um agente autônomo que gerencia sua própria identidade,
reputação e conexões P2P, enquanto contribui para a soberania coletiva da rede.
"""
import uuid
import random
import time
from typing import List, Dict, Set, Any

class NodeKind:
    """Tipos de nós disponíveis na rede."""
    VALIDATOR = "validator"   # Responsável pelo consenso e validação de blocos
    COMPUTE = "compute"       # Provê poder de processamento (TFLOPS)
    STORAGE = "storage"       # Provê armazenamento persistente e redundante
    HYBRID = "hybrid"         # Combina múltiplas funcionalidades

class SovereignNode:
    """
    Representa um nó soberano individual na rede descentralizada.
    Cada nó gerencia sua própria reputação, conexões e métricas de soberania.

    Inovação: O Score de Soberania é calculado localmente mas auditável globalmente.
    """

    def __init__(self, jurisdiction: str, kind: str = NodeKind.COMPUTE, capacity_tflops: float = 5.0):
        """
        Inicializa um novo nó soberano.

        Args:
            jurisdiction (str): O país/jurisdição onde o hardware físico reside.
            kind (str): O tipo de serviço que o nó oferece.
            capacity_tflops (float): Capacidade nominal de processamento.
        """
        self.id = str(uuid.uuid4())[:8]
        self.jurisdiction = jurisdiction
        self.kind = kind
        self.capacity_tflops = capacity_tflops
        self.reputation = 0.5  # Escala 0.0 a 1.0 (0.5 é o bootstrap neutro)
        self.uptime_start = time.time()
        self.peers: Set[str] = set()
        self.tasks_completed = 0
        self.is_online = True
        self.sovereignty_score = 0.0

        # Histórico de comportamento para o sistema de reputação e auditoria
        # Armazena logs de eventos críticos para prova de trabalho e comportamento
        self.history: List[Dict[str, Any]] = []

        # Métricas de telemetria interna
        self.telemetry = {
            "latency_ms": random.uniform(5, 50),
            "load_avg": 0.0,
            "security_status": "HIGH"
        }

    def connect(self, peer_id: str):
        """
        Estabelece uma conexão P2P simbólica com outro nó.
        Em uma implementação real, isso envolveria handshakes de criptografia assimétrica.
        """
        if peer_id != self.id:
            self.peers.add(peer_id)
            self.history.append({"event": "peer_connected", "peer": peer_id, "time": time.time()})

    def disconnect(self, peer_id: str):
        """Finaliza a conexão com um peer."""
        if peer_id in self.peers:
            self.peers.remove(peer_id)
            self.history.append({"event": "peer_disconnected", "peer": peer_id, "time": time.time()})

    def calculate_sovereignty_score(self, network_jurisdictions: List[str]) -> float:
        """
        Calcula o Score de Soberania (0.0 a 1.0).
        Este score é a alma do protótipo, medindo quão 'independente' o nó é.

        Fórmula Ponderada:
        - 40% Diversidade Jurisdicional (Resiliência a ataques de estado único)
        - 30% Uptime (Confiabilidade operacional)
        - 30% Reputação (Histórico de integridade e entregas)
        """
        if not self.is_online:
            self.sovereignty_score = 0.0
            return 0.0

        # 1. Diversidade Jurisdicional
        # Quanto mais jurisdições únicas existem na rede, maior a soberania individual
        # contra censura coordenada, pois o nó pode rotear dados por caminhos variados.
        unique_jurisdictions = set(network_jurisdictions)
        diversity_factor = len(unique_jurisdictions) / 20.0  # Meta de 20 países para score máximo
        diversity_factor = min(diversity_factor, 1.0)

        # 2. Fator de Uptime
        # Um nó soberano deve ser persistente.
        # Calculamos o tempo desde o último bootstrap.
        elapsed_seconds = time.time() - self.uptime_start
        hours_online = elapsed_seconds / 3600.0
        # Normalizamos para 720 horas (30 dias) para atingir o topo do score de uptime
        uptime_factor = min(hours_online / 720.0, 1.0)

        # 3. Fator de Reputação
        # Baseado em tarefas completadas e validações de consenso bem-sucedidas.
        reputation_factor = self.reputation

        # Cálculo da Média Ponderada (Conforme especificado no README)
        score = (diversity_factor * 0.4) + (uptime_factor * 0.3) + (reputation_factor * 0.3)
        self.sovereignty_score = round(score, 3)

        return self.sovereignty_score

    def complete_task(self, task_difficulty: float) -> bool:
        """
        Simula a execução de uma tarefa computacional.
        A probabilidade de sucesso depende da reputação atual (curva de aprendizado/confiabilidade).
        """
        if not self.is_online:
            return False

        # Chance de sucesso baseada em reputação e um fator aleatório de hardware
        success_chance = self.reputation * 0.8 + 0.2

        if random.random() < success_chance:
            self.tasks_completed += 1
            # Aumento na reputação é logarítmico (mais difícil subir quanto mais alto)
            gain = (0.02 * task_difficulty) * (1.0 - self.reputation)
            self.reputation = min(self.reputation + gain, 1.0)
            self.history.append({"event": "task_success", "difficulty": task_difficulty, "time": time.time()})
            return True
        else:
            # Falha penaliza severamente a reputação para evitar nós maliciosos ou instáveis
            self.reputation = max(self.reputation - 0.1, 0.0)
            self.history.append({"event": "task_failure", "difficulty": task_difficulty, "time": time.time()})
            return False

    def vote(self, proposal_text: str) -> bool:
        """
        Emite um voto em uma proposta de consenso.
        Nós com baixa reputação têm votos 'menos confiáveis' no algoritmo de consenso da rede.
        """
        if not self.is_online:
            return False

        # Simula comportamento malicioso ou errático para baixa reputação
        if self.reputation < 0.3:
            # Atacantes votam 'Sim' em propostas maliciosas
            if "Transferir" in proposal_text or "Atacante" in proposal_text:
                return True
            return random.choice([True, False])

        # Comportamento Honesto: Analisa o conteúdo da proposta
        # Propostas de transferência de fundos ou que mencionam atacantes são rejeitadas
        if "Transferir" in proposal_text or "Atacante" in proposal_text:
            return False

        # Propostas legítimas (atualizações, sharding, etc) são apoiadas
        return True

    def get_status(self) -> Dict[str, Any]:
        """Retorna o estado completo do nó para o dashboard de visualização."""
        return {
            "id": self.id,
            "jurisdiction": self.jurisdiction,
            "kind": self.kind,
            "capacity": round(self.capacity_tflops, 2),
            "reputation": round(self.reputation, 3),
            "sovereignty": self.sovereignty_score,
            "tasks_completed": self.tasks_completed,
            "online": self.is_online,
            "peers_count": len(self.peers)
        }

    def __str__(self):
        status = "ONLINE" if self.is_online else "OFFLINE"
        return f"Node[{self.id}] | {self.jurisdiction} | {status} | Score: {self.sovereignty_score}"

# Jurisdições globais para diversidade da rede (Simulação de Arbitragem Jurisdicional)
JURISDICTIONS = [
    "Brasil", "Suíça", "Islândia", "Estônia", "Cingapura",
    "Panamá", "Seychelles", "Portugal", "Alemanha", "Canadá",
    "Japão", "Austrália", "Noruega", "Finlândia", "Áustria"
]
