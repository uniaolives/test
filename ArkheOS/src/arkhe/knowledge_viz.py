# arkhe/knowledge_viz.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from .cortex_memory import CortexMemory
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import json

class ArkheViz:
    """
    Motor de visualização da gravidade semântica (v5.0).
    Implementa análise topológica do hipergrafo de conhecimento.
    """
    def __init__(self, cortex: CortexMemory):
        self.cortex = cortex
        self.topology_data = None

    def analyze_topology(self, perplexity: int = 30) -> Dict[str, Any]:
        """
        Análise completa da topologia do conhecimento.
        """
        print("🔭 Analisando topologia do córtex...")

        # 1. Extração de dados
        data = self.cortex.collection.get(include=['embeddings', 'metadatas', 'documents'])
        ids = data['ids']
        if not ids or len(ids) < 3:
            print("⚠️ Memória insuficiente para análise topológica.")
            return {}

        embeddings = np.array(data['embeddings'])
        metadatas = data['metadatas']

        # 2. Redução dimensional (t-SNE)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(ids)-1),
            init='pca',
            learning_rate='auto',
            random_state=42
        )
        coords_2d = tsne.fit_transform(embeddings)

        # 3. Construção do grafo semântico
        G = nx.Graph()
        for i, node_id in enumerate(ids):
            topic = metadatas[i].get('topic', 'Unknown')
            G.add_node(node_id, pos=coords_2d[i], topic=topic)

            # Arestas por similaridade de cosseno (Threshold)
            for j in range(i + 1, len(ids)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                if sim > 0.7:
                    G.add_edge(node_id, ids[j], weight=sim)

        # 4. Detecção de clusters (modularity)
        communities = list(nx.community.greedy_modularity_communities(G))
        cluster_map = {}
        for i, community in enumerate(communities):
            for node in community:
                cluster_map[node] = i

        # 5. Cálculo de densidade e pontes
        centrality = nx.degree_centrality(G)

        bridges = []
        for edge in G.edges():
            n1, n2 = edge
            c1, c2 = cluster_map.get(n1), cluster_map.get(n2)
            if c1 != c2:
                bridges.append({
                    "source": n1, "target": n2,
                    "clusters": (c1, c2),
                    "weight": G[n1][n2]['weight']
                })

        self.topology_data = {
            "G": G,
            "coords": coords_2d,
            "cluster_map": cluster_map,
            "communities": communities,
            "centrality": centrality,
            "bridges": bridges,
            "coherence_global": np.mean([m.get('confidence', 0.5) for m in metadatas])
        }

        print(f"   📊 Análise completa: {len(ids)} nós, {len(communities)} clusters, {len(bridges)} pontes.")
        return self.topology_data

    def visualize(self, save_path: str = "arkhe_topology.png"):
        """Gera a visualização topológica v5.0."""
        if not self.topology_data:
            self.analyze_topology()

        if not self.topology_data: return

        G = self.topology_data["G"]
        pos = nx.get_node_attributes(G, 'pos')
        cluster_map = self.topology_data["cluster_map"]

        plt.figure(figsize=(15, 10))

        # Desenhar nós coloridos por cluster
        colors = [cluster_map.get(node, 0) for node in G.nodes()]
        node_sizes = [self.topology_data["centrality"][node] * 2000 + 100 for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors,
                               cmap=plt.cm.tab20, alpha=0.8)

        # Desenhar arestas normais (intra-cluster)
        intra_edges = [e for e in G.edges() if cluster_map[e[0]] == cluster_map[e[1]]]
        nx.draw_networkx_edges(G, pos, edgelist=intra_edges, alpha=0.2, edge_color='gray')

        # Desenhar pontes (inter-cluster)
        inter_edges = [e for e in G.edges() if cluster_map[e[0]] != cluster_map[e[1]]]
        nx.draw_networkx_edges(G, pos, edgelist=inter_edges, alpha=0.6,
                               edge_color='red', style='dashed', width=1.5)

        # Labels para os centros de cada cluster
        for i, community in enumerate(self.topology_data["communities"]):
            # Encontrar o nó mais central na comunidade
            best_node = max(community, key=lambda n: self.topology_data["centrality"][n])
            plt.text(pos[best_node][0], pos[best_node][1], G.nodes[best_node]['topic'],
                     fontsize=10, fontweight='bold', ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.title(f"Arkhe(n) v5.0 - Topologia do Conhecimento (C_global={self.topology_data['coherence_global']:.2f})")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualização salva em: {save_path}")

        plt.close()

if __name__ == "__main__":
    memory = CortexMemory()
    # Adicionar alguns dados se estiver vazio para o teste
    if memory.collection.count() < 3:
        memory.memorize("RFID", "Rastreabilidade física via rádio.", 0.95, "doc1", ["IoT", "Hardware"])
        memory.memorize("IoT", "Internet das Coisas e conexão de nós.", 0.92, "doc2", ["RFID", "Redes"])
        memory.memorize("Arkhe", "Unificação de sistemas via x2 = x + 1.", 0.99, "doc3", ["Ontologia"])
        memory.memorize("Ontologia", "Estudo do ser e da identidade.", 0.90, "doc4", ["Arkhe"])
        memory.memorize("Física", "Leis que regem o mundo material.", 0.88, "doc5", ["Mecânica"])
        memory.memorize("Mecânica", "Estudo do movimento dos corpos.", 0.85, "doc6", ["Física"])

    viz = ArkheViz(memory)
    viz.visualize("test_v5_topology.png")
