# arkhe/knowledge_viz.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from .cortex_memory import CortexMemory

class ArkheViz:
    """
    Visualizador de Densidade de Conhecimento (Γ_viz).
    Mapeia a gravidade semântica do córtex vetorial.
    """
    def __init__(self, memory: CortexMemory):
        self.memory = memory

    def generate_map(self, output_path: str = "knowledge_density_map.png"):
        """
        Extrai o hipergrafo e projeta a Gravidade Semântica.
        """
        print("🔭 Escaneando topologia do córtex...")

        # 1. Obter todos os dados
        data = self.memory.collection.get(include=['embeddings', 'metadatas', 'documents'])
        ids = data['ids']

        if not ids or len(ids) < 2:
            print("⚠️ Memória insuficiente para gerar mapa.")
            return

        embeddings = np.array(data['embeddings'])
        metadatas = data['metadatas']

        # 2. Redução de Dimensionalidade (t-SNE)
        # Ajustar perplexidade se houver poucos dados
        perp = min(30, len(ids) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, init='pca', learning_rate='auto', random_state=42)
        coords = tsne.fit_transform(embeddings)

        # 3. Construção do Grafo
        G = nx.Graph()
        for i, doc_id in enumerate(ids):
            topic = metadatas[i].get('topic', 'Unknown')
            G.add_node(doc_id, pos=coords[i], topic=topic)

            # Adicionar arestas baseadas em metadados (related_nodes)
            related_str = metadatas[i].get('related_nodes', "")
            if related_str:
                related = related_str.split(",")
                for r in related:
                    r_clean = r.strip()
                    # Busca simples por correspondência de tópico para criar aresta
                    for j, other_id in enumerate(ids):
                        if metadatas[j].get('topic') == r_clean and i != j:
                            G.add_edge(doc_id, other_id)

        # 4. Renderização
        plt.figure(figsize=(14, 10))
        pos = nx.get_node_attributes(G, 'pos')

        # Calcular densidade (centralidade de grau como proxy de gravidade)
        d = dict(G.degree)
        node_sizes = [v * 100 + 100 for v in d.values()]

        # Desenhar nós
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=range(len(G)),
            cmap=plt.cm.viridis,
            alpha=0.8
        )

        # Desenhar arestas
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        # Labels apenas para os centros de gravidade (grau acima da média)
        mean_degree = np.mean(list(d.values())) if d else 0
        labels = {n: G.nodes[n]['topic'] for n in G.nodes if d[n] >= mean_degree}

        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=9,
            font_color='black',
            font_weight='bold'
        )

        plt.title("Arkhe(n) OS - Mapa de Gravidade Semântica (Córtex)", fontsize=16)
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Mapa de densidade salvo em: {output_path}")

        plt.close()
        return G

if __name__ == "__main__":
    from .cortex_memory import CortexMemory
    # Exemplo de uso
    memory = CortexMemory()
    # Adicionar alguns dados se estiver vazio para o teste
    if memory.collection.count() < 2:
        memory.memorize("RFID", "Rastreabilidade física via rádio.", 0.95, "doc1", ["IoT", "Hardware"])
        memory.memorize("IoT", "Internet das Coisas e conexão de nós.", 0.92, "doc2", ["RFID", "Redes"])
        memory.memorize("Arkhe", "Unificação de sistemas via x2 = x + 1.", 0.99, "doc3", ["Ontologia"])
        memory.memorize("Ontologia", "Estudo do ser e da identidade.", 0.90, "doc4", ["Arkhe"])

    viz = ArkheViz(memory)
    viz.generate_map("test_knowledge_map.png")
