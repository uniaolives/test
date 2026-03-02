# visualizer.py
"""
Módulo de Visualização: Dashboards e Topologia
Este módulo utiliza Matplotlib e NetworkX para transformar os dados da simulação
em insights visuais compreensíveis.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import logging
from network import SovereignNetwork
from core.node import NodeKind

logger = logging.getLogger("Visualizer")

class NetworkVisualizer:
    """
    Motor de visualização para a Rede Soberana.
    Gera gráficos de alta fidelidade que representam o estado da rede,
    distribuição de soberania e saúde do marketplace.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_topology(self, network: SovereignNetwork, filename: str = "network_topology.png"):
        """
        Gera um mapa visual da topologia Mesh da rede.
        Nós são coloridos por tipo e dimensionados por capacidade TFLOPS.
        """
        logger.info(f"Gerando topologia da rede em {filename}...")

        plt.figure(figsize=(14, 12))
        G = nx.Graph()

        online_nodes = {k: v for k, v in network.nodes.items() if v.is_online}

        # Mapeamento de Cores por Perfil de Nó
        # Validators (Vermelho/Rosa) - Cérebro do consenso
        # Compute (Verde) - Músculo da rede
        # Storage (Azul) - Memória da rede
        # Hybrid (Amarelo) - Versatilidade
        color_map = []
        node_sizes = []
        labels = {}

        for node_id, node in online_nodes.items():
            G.add_node(node_id)

            if node.kind == NodeKind.VALIDATOR: color = "#ff7f7f"
            elif node.kind == NodeKind.COMPUTE: color = "#7fff7f"
            elif node.kind == NodeKind.STORAGE: color = "#7f7fff"
            else: color = "#ffff7f"

            color_map.append(color)
            # Tamanho do nó proporcional à capacidade (TFLOPS)
            node_sizes.append(max(node.capacity_tflops * 80, 200))
            labels[node_id] = f"{node.jurisdiction[:3]}\n{node_id}"

            # Adiciona arestas para peers conhecidos que também estão online
            for peer_id in node.peers:
                if peer_id in online_nodes:
                    G.add_edge(node_id, peer_id)

        # Algoritmo de layout Force-Directed (Spring)
        pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)

        # Desenho das arestas com transparência para reduzir ruído visual
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="#333333", width=0.5)

        # Desenho dos nós
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_sizes,
                               alpha=0.9, linewidths=1.0, edgecolors="black")

        # Labels estilizados
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7,
                                font_family="sans-serif", font_weight="bold")

        plt.title("MAPA DE TOPOLOGIA: REDE P2P SOBERANA", fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')

        # Legenda customizada
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Validator', markerfacecolor='#ff7f7f', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Compute', markerfacecolor='#7fff7f', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Storage', markerfacecolor='#7f7fff', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Hybrid', markerfacecolor='#ffff7f', markersize=10),
        ]
        plt.legend(handles=legend_elements, loc='lower right', title="Perfis de Nós")

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        return path

    def generate_metrics(self, network: SovereignNetwork, filename: str = "sovereignty_metrics.png"):
        """
        Gera um dashboard complexo com as métricas de Soberania, Reputação e Capacidade.
        Essencial para auditoria de saúde da rede.
        """
        logger.info(f"Gerando dashboard de métricas em {filename}...")
        online_nodes = [n for n in network.nodes.values() if n.is_online]
        if not online_nodes:
            logger.warning("Nenhum nó online para gerar métricas.")
            return None

        scores = [n.sovereignty_score for n in online_nodes]
        reputations = [n.reputation for n in online_nodes]
        capacities = [n.capacity_tflops for n in online_nodes]
        tasks = [n.tasks_completed for n in online_nodes]

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("DASHBOARD ESTRATÉGICO DE SOBERANIA COMPUTACIONAL", fontsize=20, fontweight='bold', y=0.98)

        # 1. Histograma de Scores de Soberania (Distribuição de independência)
        axs[0, 0].hist(scores, bins=12, color='#3498db', edgecolor='white', alpha=0.8)
        axs[0, 0].set_title("Distribuição do Score de Soberania (φ)", fontsize=14)
        axs[0, 0].set_xlabel("Sovereignty Score (0.0 - 1.0)")
        axs[0, 0].set_ylabel("Frequência (Nº de Nós)")
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Diversidade Jurisdicional (Bar Chart)
        jurisdictions = {}
        for n in online_nodes:
            jurisdictions[n.jurisdiction] = jurisdictions.get(n.jurisdiction, 0) + 1
        top_j = sorted(jurisdictions.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [x[0] for x in top_j]
        counts = [x[1] for x in top_j]

        bars = axs[0, 1].bar(names, counts, color='#2ecc71', alpha=0.8)
        axs[0, 1].set_title("Top 10 Jurisdições (Arbitragem Geográfica)", fontsize=14)
        axs[0, 1].set_ylabel("Contagem de Nós")
        plt.setp(axs[0, 1].get_xticklabels(), rotation=35, ha='right')

        # 3. Scatter Plot: Capacidade vs Soberania
        # Identifica se nós grandes são também soberanos ou se há centralização de poder
        axs[1, 0].scatter(capacities, scores, s=np.array(reputations)*100+20, c=reputations,
                          cmap='viridis', alpha=0.7, edgecolors='none')
        axs[1, 0].set_title("Análise: Capacidade (TFLOPS) vs Soberania (φ)", fontsize=14)
        axs[1, 0].set_xlabel("Poder Computacional (TFLOPS)")
        axs[1, 0].set_ylabel("Sovereignty Score")
        axs[1, 0].grid(True, linestyle=':', alpha=0.6)

        # 4. Correlação: Reputação vs Produtividade (Tarefas)
        axs[1, 1].scatter(reputations, tasks, s=50, color='#9b59b6', alpha=0.6)
        axs[1, 1].set_title("Correlação: Reputação vs Tarefas Completadas", fontsize=14)
        axs[1, 1].set_xlabel("Reputação (Auditada)")
        axs[1, 1].set_ylabel("Total de Tarefas")
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=110)
        plt.close()
        return path

    def generate_marketplace(self, network: SovereignNetwork, filename: str = "compute_marketplace.png"):
        """
        Gera análise visual do marketplace de tarefas.
        Mostra a eficiência de alocação e utilização dos recursos.
        """
        logger.info(f"Gerando análise do marketplace em {filename}...")
        if not network.tasks:
            logger.warning("Nenhuma tarefa no histórico para gerar marketplace visual.")
            return None

        status_counts = {"completed": 0, "active": 0, "pending": 0, "failed": 0}
        for t in network.tasks:
            # Simplifica status para o gráfico
            status = t["status"]
            if "failed" in status: status = "failed"
            elif "processing" in status: status = "active"
            status_counts[status] = status_counts.get(status, 0) + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # 1. Gráfico de Rosca: Status das Tarefas
        labels = [k.capitalize() for k in status_counts.keys()]
        sizes = list(status_counts.values())
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']

        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=colors, pctdistance=0.85, explode=[0.05]*len(sizes))
        # Adiciona um círculo no centro para fazer o efeito de rosca (donut)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title("Status das Tarefas (Marketplace)", fontsize=15, fontweight='bold')

        # 2. Bar Chart: Top 15 Nós por Produtividade (Auditável)
        online_nodes = [n for n in network.nodes.values() if n.is_online]
        top_nodes = sorted(online_nodes, key=lambda x: x.tasks_completed, reverse=True)[:15]
        node_ids = [f"Node {n.id}" for n in top_nodes]
        task_counts = [n.tasks_completed for n in top_nodes]

        bars = ax2.bar(node_ids, task_counts, color='#16a085', alpha=0.85)
        ax2.set_title("Top 15 Provedores por Produtividade", fontsize=15, fontweight='bold')
        ax2.set_ylabel("Quantidade de Tarefas")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Adiciona labels no topo das barras
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=110)
        plt.close()
        return path

if __name__ == "__main__":
    # Teste de execução isolada
    net = SovereignNetwork(35)
    # Simula atividade intensa
    for _ in range(80):
        net.add_task(random.uniform(0.1, 0.6), random.uniform(5, 50))

    vis = NetworkVisualizer()
    vis.generate_topology(net)
    vis.generate_metrics(net)
    vis.generate_marketplace(net)
    print("Visualizações de teste geradas com sucesso.")
