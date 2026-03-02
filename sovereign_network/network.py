# network.py
"""
Módulo de Rede: Orquestração e Consenso
Este módulo implementa a lógica de malha P2P, o algoritmo de Consenso Bizantino
Ponderado por Reputação e o Marketplace de Computação.
"""
import random
import time
import logging
from typing import List, Dict, Optional, Any
from core.node import SovereignNode, JURISDICTIONS, NodeKind

# Configuração de Logging para auditoria da rede
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("SovereignNetwork")

class SovereignNetwork:
    """
    Representa a rede descentralizada completa.
    Funciona como o "plano de controle" simulado para a malha P2P.

    Inovação: Consenso Bizantino onde o poder de voto não é 'um nó, um voto',
    mas sim ponderado pela reputação acumulada e pelo Score de Soberania.
    """

    def __init__(self, node_count: int = 25):
        """
        Inicializa a rede.

        Args:
            node_count (int): Número inicial de nós para o bootstrap.
        """
        self.nodes: Dict[str, SovereignNode] = {}
        self.proposals: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.genesis_time = time.time()

        logger.info(f"Iniciando Bootstrap da Rede Soberana com {node_count} nós...")
        self.bootstrap(node_count)

    def bootstrap(self, count: int):
        """
        Cria os nós iniciais e estabelece a topologia Mesh P2P.
        Garante diversidade jurisdicional desde o bloco gênese.
        """
        # 1. Criação de nós com diversidade geográfica
        for _ in range(count):
            jurisdiction = random.choice(JURISDICTIONS)
            # Distribuição de tipos de nós
            kind_rand = random.random()
            if kind_rand < 0.2: kind = NodeKind.VALIDATOR
            elif kind_rand < 0.6: kind = NodeKind.COMPUTE
            elif kind_rand < 0.8: kind = NodeKind.STORAGE
            else: kind = NodeKind.HYBRID

            capacity = random.uniform(2.0, 15.0)
            node = SovereignNode(jurisdiction, kind, capacity)
            self.nodes[node.id] = node

        # 2. Criação da Malha P2P (Topology Construction)
        # Cada nó tenta se conectar a um subconjunto aleatório de outros nós
        all_ids = list(self.nodes.keys())
        for node in self.nodes.values():
            target_connections = random.randint(4, 10)
            # Seleciona peers aleatórios excluindo a si mesmo
            potential_peers = random.sample(all_ids, min(target_connections + 1, len(all_ids)))
            for peer_id in potential_peers:
                if peer_id != node.id:
                    node.connect(peer_id)

        # 3. Cálculo inicial de scores
        self.update_all_scores()
        logger.info("Bootstrap concluído. Topologia Mesh estabelecida.")

    def update_all_scores(self):
        """Recalcula métricas de soberania para toda a rede."""
        # Coleta lista de jurisdições online para cálculo de diversidade
        current_jurisdictions = [n.jurisdiction for n in self.nodes.values() if n.is_online]
        for node in self.nodes.values():
            node.calculate_sovereignty_score(current_jurisdictions)

    def run_consensus(self, proposal_text: str) -> bool:
        """
        Executa o Algoritmo de Consenso Bizantino Ponderado.

        Regra: Para aprovação, a soma das reputações dos nós que votaram 'Sim'
        deve ser superior a 67% da soma total das reputações de todos os nós online.
        Isso protege contra ataques Sybil, onde um atacante cria muitos nós com
        baixa reputação.
        """
        proposal_id = f"PROP-{len(self.proposals):03d}"
        logger.info(f"Nova Proposta de Consenso: {proposal_id} - '{proposal_text}'")

        votes_pro_weight = 0.0
        total_online_weight = 0.0

        online_nodes = [n for n in self.nodes.values() if n.is_online]

        for node in online_nodes:
            # O peso do voto é a reputação do nó
            weight = node.reputation
            total_online_weight += weight

            # Nós honestos analisam o texto da proposta
            if node.vote(proposal_text):
                votes_pro_weight += weight

        approved = False
        participation_ratio = votes_pro_weight / total_online_weight if total_online_weight > 0 else 0

        if participation_ratio > 0.67:
            approved = True
            logger.info(f"Proposta {proposal_id} APROVADA com {participation_ratio*100:.1f}% de peso.")
        else:
            logger.warning(f"Proposta {proposal_id} REJEITADA. Apenas {participation_ratio*100:.1f}% de suporte.")

        self.proposals.append({
            "id": proposal_id,
            "text": proposal_text,
            "approved": approved,
            "weight_support": participation_ratio,
            "timestamp": time.time()
        })
        return approved

    def add_task(self, difficulty: float, reward: float) -> bool:
        """
        Adiciona uma tarefa computacional ao marketplace.
        A rede aloca a tarefa ao nó mais apto disponível.
        """
        task_id = f"TASK-{len(self.tasks):04d}"
        task_entry = {
            "id": task_id,
            "difficulty": difficulty,
            "reward": reward,
            "status": "pending",
            "assigned_to": None
        }
        self.tasks.append(task_entry)

        # Seleção de Provedor (Marketplace Auction Simulation)
        # Filtramos nós online que possuem capacidade computacional
        candidates = [n for n in self.nodes.values() if n.is_online and n.kind in [NodeKind.COMPUTE, NodeKind.HYBRID]]

        if not candidates:
            logger.error(f"Falha ao alocar {task_id}: Nenhum nó de computação disponível.")
            task_entry["status"] = "failed_no_provider"
            return False

        # O "leilão" escolhe o nó com o melhor equilíbrio entre reputação e capacidade
        candidates.sort(key=lambda x: (x.reputation * 0.7 + (x.capacity_tflops/15.0) * 0.3), reverse=True)
        winner = candidates[0]

        task_entry["assigned_to"] = winner.id
        task_entry["status"] = "processing"

        # Simula o processamento
        if winner.complete_task(difficulty):
            task_entry["status"] = "completed"
            return True
        else:
            task_entry["status"] = "failed_execution"
            return False

    def simulate_censorship(self, target_jurisdiction: str) -> int:
        """
        Simula uma interrupção em massa em uma jurisdição específica.
        Útil para testar a resiliência da rede e a variação do Score de Soberania.
        """
        logger.warning(f"ALERTA DE SEGURANÇA: Simulando censura governamental em '{target_jurisdiction}'...")
        affected_count = 0
        for node in self.nodes.values():
            if node.jurisdiction == target_jurisdiction and node.is_online:
                node.is_online = False
                affected_count += 1

        self.update_all_scores()
        logger.info(f"Ataque concluído. {affected_count} nós foram desconectados forçadamente.")
        return affected_count

    def simulate_sybil_attack(self, count: int) -> List[str]:
        """
        Simula um ataque Sybil injetando múltiplos nós maliciosos na rede.
        Esses nós terão reputação inicial baixa e tentarão subverter o consenso.
        """
        logger.warning(f"ALERTA DE SEGURANÇA: Detectada tentativa de injeção Sybil ({count} nós)...")
        sybil_ids = []
        for i in range(count):
            # Atacante cria nós com configurações mínimas
            node = SovereignNode("Malicious_Zone", NodeKind.VALIDATOR, 1.0)
            node.id = f"sybil_{i:03d}_{random.randint(1000,9999)}"
            node.reputation = 0.05 # Reputação baixíssima de entrada
            self.nodes[node.id] = node
            sybil_ids.append(node.id)

        self.update_all_scores()
        return sybil_ids

    def get_network_metrics(self) -> Dict[str, Any]:
        """Calcula métricas agregadas de saúde e soberania da rede."""
        online_nodes = [n for n in self.nodes.values() if n.is_online]
        if not online_nodes:
            return {"status": "OFFLINE"}

        total_capacity = sum(n.capacity_tflops for n in online_nodes)
        avg_sovereignty = sum(n.sovereignty_score for n in online_nodes) / len(online_nodes)
        avg_reputation = sum(n.reputation for n in online_nodes) / len(online_nodes)

        juris_map = {}
        for n in online_nodes:
            juris_map[n.jurisdiction] = juris_map.get(n.jurisdiction, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "online_nodes": len(online_nodes),
            "total_capacity_tflops": round(total_capacity, 2),
            "avg_sovereignty": round(avg_sovereignty, 3),
            "avg_reputation": round(avg_reputation, 3),
            "jurisdiction_diversity": len(juris_map),
            "top_jurisdictions": sorted(juris_map.items(), key=lambda x: x[1], reverse=True)[:5]
        }

if __name__ == "__main__":
    # Teste de fumaça
    net = SovereignNetwork(30)
    print(f"Métricas Iniciais: {net.get_network_metrics()}")

    # Simula carga de trabalho
    for _ in range(20):
        net.add_task(random.uniform(0.1, 0.9), 15.0)

    # Testa consenso
    net.run_consensus("Migrar Kernel para Proteção Quântica")

    print(f"Métricas Finais: {net.get_network_metrics()}")
