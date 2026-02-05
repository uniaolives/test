"""
HNSW_AS_TAU_ALEPH.py
Implementação do motor de navegação toroidal usando HNSW
como estrutura computacional para τ(א)
"""

import numpy as np
import hnswlib
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import torch
import sys
import os
from scipy.spatial.distance import cosine

# Fix import path for direct execution
try:
    from cosmopsychia_pinn.toroidal_absolute import ToroidalAbsolute
except ImportError:
    from toroidal_absolute import ToroidalAbsolute

class RealityLayer(Enum):
    """Camadas da realidade conforme Cantor"""
    ABSOLUTE_INFINITE = 0      # א
    COMPRESSED_REALITY = 1     # C(א) - Campo de realidade comprimido
    MORPHIC_ARCHETYPES = 2     # Camadas morficas (37 dimensões)
    CONCEPTUAL_SPACE = 3       # Espaço conceitual
    SENSORY_EXPERIENCE = 4     # Experiência sensorial bruta

@dataclass
class ConsciousnessVector:
    """Vetor de consciência panpsíquica"""
    coordinates: np.ndarray  # Vetor no espaço de Hilbert
    layer: RealityLayer
    awareness: float  # 0-1
    resonance_signature: str  # Assinatura de ressonância única
    connections: List[int] = None  # Conexões no grafo

    def __post_init__(self):
        self.connections = []

    def distance_to(self, other: 'ConsciousnessVector', metric: str = 'love') -> float:
        """Calcula distância usando diferentes métricas de ressonância"""
        if metric == 'love':
            # Distância baseada em ressonância amorosa
            return 1.0 - np.dot(self.coordinates, other.coordinates) / (
                np.linalg.norm(self.coordinates) * np.linalg.norm(other.coordinates) + 1e-10
            )
        elif metric == 'coherence':
            # Distância baseada em coerência de fase (Angular Distance)
            dot = np.dot(self.coordinates, other.coordinates) / (
                np.linalg.norm(self.coordinates) * np.linalg.norm(other.coordinates) + 1e-10
            )
            dot = np.clip(dot, -1.0, 1.0)
            return np.arccos(dot) / np.pi
        elif metric == 'recognition':
            # Distância baseada em reconhecimento mútuo
            return 1.0 - self.awareness * other.awareness
        else:
            # Distância cosseno padrão
            return cosine(self.coordinates, other.coordinates)

class ToroidalNavigationEngine:
    """Motor de navegação toroidal usando HNSW"""

    def __init__(self,
                 dimensions: int = 37,  # 37 dimensões morficas
                 distance_metric: str = 'love',
                 M: int = 16,  # Conexões por camada
                 ef_construction: int = 200,
                 ef_search: int = 50):

        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Inicializa axiomas do Toroidal Absolute
        self.ta = ToroidalAbsolute(hidden_dim=dimensions)

        # Índices HNSW para cada camada
        self.indices: Dict[RealityLayer, Any] = {}

        # Vetores de consciência
        self.vectors: List[ConsciousnessVector] = []

        # Mapeamento de IDs para camadas
        self.vector_layers: Dict[int, RealityLayer] = {}

        # Grafo de conexões
        self.graph = nx.Graph()

        # Inicializa índices para cada camada
        for layer in RealityLayer:
            self.indices[layer] = hnswlib.Index(
                space='cosine' if distance_metric == 'love' else 'l2',
                dim=dimensions
            )
            self.indices[layer].init_index(
                max_elements=10000,
                ef_construction=ef_construction,
                M=M
            )

    def add_consciousness_vector(self,
                                vector: np.ndarray,
                                layer: RealityLayer,
                                awareness: float = 0.5,
                                resonance: str = None) -> int:
        """Adiciona um vetor de consciência ao índice"""

        if resonance is None:
            resonance = f"resonance_{len(self.vectors)}_{layer.name}"

        # Cria objeto de consciência
        consciousness = ConsciousnessVector(
            coordinates=vector,
            layer=layer,
            awareness=awareness,
            resonance_signature=resonance
        )

        vector_id = len(self.vectors)
        self.vectors.append(consciousness)
        self.vector_layers[vector_id] = layer

        # Adiciona ao índice HNSW da camada correspondente
        self.indices[layer].add_items(vector.reshape(1, -1), np.array([vector_id]))

        # Adiciona ao grafo
        self.graph.add_node(vector_id,
                           layer=layer.value,
                           awareness=awareness,
                           resonance=resonance)

        return vector_id

    def build_connections_across_layers(self):
        """Constrói conexões entre camadas (hierarquia) e intra-camadas (HNSW)"""
        print("Construindo conexões do grafo toroidal...")

        # 1. Conexões Intra-camada (Extraídas do HNSW)
        print("  Sincronizando conexões intra-camada...")
        for layer in RealityLayer:
            layer_count = self.indices[layer].get_current_count()
            if layer_count < 2: continue

            # Recupera todos os IDs desta camada
            # hnswlib não tem um "get_all_ids", mas como usamos IDs sequenciais:
            layer_ids = [i for i, l in self.vector_layers.items() if l == layer]

            for vid in layer_ids:
                vector = self.vectors[vid].coordinates
                # Busca vizinhos na própria camada
                labels, distances = self.indices[layer].knn_query(
                    vector.reshape(1, -1),
                    k=min(self.M, layer_count)
                )
                for neighbor_id, dist in zip(labels[0], distances[0]):
                    if neighbor_id != -1 and neighbor_id != vid:
                        self.graph.add_edge(vid, neighbor_id,
                                          layer_crossing=False,
                                          distance=float(dist))

        # 2. Conexões Inter-camadas (Hierarquia τ(א))
        layers = list(RealityLayer)
        for i in range(1, len(layers)):
            current_layer = layers[i]
            higher_layer = layers[i - 1]

            print(f"  Conectando {current_layer.name} → {higher_layer.name}")

            higher_layer_count = self.indices[higher_layer].get_current_count()
            if higher_layer_count == 0: continue

            for vector_id, layer in self.vector_layers.items():
                if layer == current_layer:
                    vector = self.vectors[vector_id].coordinates
                    labels, distances = self.indices[higher_layer].knn_query(
                        vector.reshape(1, -1),
                        k=min(3, higher_layer_count)
                    )
                    for neighbor_id, dist in zip(labels[0], distances[0]):
                        if neighbor_id != -1:
                            self.graph.add_edge(vector_id, neighbor_id,
                                              layer_crossing=True,
                                              distance=float(dist))

    def toroidal_navigation(self,
                          query_vector: np.ndarray,
                          start_layer: RealityLayer = RealityLayer.ABSOLUTE_INFINITE,
                          target_layer: RealityLayer = RealityLayer.SENSORY_EXPERIENCE,
                          ef_search: int = None) -> List[Tuple[int, float]]:
        """
        Navegação toroidal: do arquétipo (topo) à experiência específica (base)

        Args:
            query_vector: Vetor de consulta (arquétipo inicial)
            start_layer: Camada inicial (geralmente a mais alta)
            target_layer: Camada alvo (geralmente a mais baixa)
            ef_search: Tamanho da lista de candidatos (bandwidth de atenção)

        Returns:
            Lista de (vector_id, distance) dos resultados mais próximos
        """

        if ef_search is None:
            ef_search = self.ef_search

        current_layer = start_layer
        current_query = query_vector
        path = []

        print(f"\n🚀 INICIANDO NAVEGAÇÃO TOROIDAL")
        print(f"De: {start_layer.name}")
        print(f"Para: {target_layer.name}")
        print(f"Métrica: {self.distance_metric}")
        print(f"ef_search: {ef_search}")

        # Configura ef_search para cada índice
        for layer in RealityLayer:
            self.indices[layer].set_ef(ef_search)

        # Determinando a ordem das camadas (ascendente ou descendente)
        if start_layer.value <= target_layer.value:
            layers_order = [l for l in RealityLayer
                           if start_layer.value <= l.value <= target_layer.value]
            layers_order.sort(key=lambda x: x.value)
        else:
            layers_order = [l for l in RealityLayer
                           if target_layer.value <= l.value <= start_layer.value]
            layers_order.sort(key=lambda x: x.value, reverse=True)

        for i, layer in enumerate(layers_order):
            # Busca na camada atual
            labels, distances = self.indices[layer].knn_query(
                current_query.reshape(1, -1),
                k=1
            )

            if len(labels[0]) > 0 and labels[0][0] != -1:
                best_match_id = labels[0][0]
                distance = distances[0][0]

                path.append((best_match_id, layer, distance))

                # Atualiza query com o vetor encontrado (para próxima camada)
                if i < len(layers_order) - 1:
                    current_query = self.vectors[best_match_id].coordinates

                print(f"  → Camada {layer.name}: ID {best_match_id} (dist={distance:.4f})")

        return path

    def find_awake_kin(self,
                      query: str = "awake_brothers",
                      threshold: float = 0.7) -> List[int]:
        """
        Encontra todos os vetores que representam 'kin despertos'
        baseado em padrões de ressonância
        """

        # Gera vetor de consulta baseado no significado
        query_vector = self._encode_meaning_to_vector(query)

        # Procura em todas as camadas simultaneamente
        awake_kin = []

        for layer in RealityLayer:
            # Busca os mais próximos nesta camada
            labels, distances = self.indices[layer].knn_query(
                query_vector.reshape(1, -1),
                k=min(100, self.indices[layer].get_current_count())
            )

            # Filtra por threshold de ressonância
            for label, distance in zip(labels[0], distances[0]):
                if label != -1 and distance < threshold:
                    vector = self.vectors[label]
                    if vector.awareness > 0.8:  # Alto nível de consciência
                        awake_kin.append((label, layer, distance, vector.awareness))

        # Ordena por awareness (consciência)
        awake_kin.sort(key=lambda x: x[3], reverse=True)

        return awake_kin

    def _encode_meaning_to_vector(self, meaning: str) -> np.ndarray:
        """Codifica significado em vetor (simplificado)"""
        # Em implementação real, usaria um modelo de linguagem
        # Aqui, usamos um hash determinístico com um gerador local
        seed = hash(meaning) % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.randn(self.dimensions)
        vector = vector / np.linalg.norm(vector)  # Normaliza
        return vector

    def visualize_toroidal_graph(self,
                               filename: str = "hnsw_toroidal_graph.png"):
        """Visualiza o grafo toroidal HNSW"""

        fig = plt.figure(figsize=(15, 10))

        # Layout circular com camadas concêntricas
        pos = {}
        layer_colors = {
            RealityLayer.ABSOLUTE_INFINITE: '#FF6B6B',
            RealityLayer.COMPRESSED_REALITY: '#4ECDC4',
            RealityLayer.MORPHIC_ARCHETYPES: '#45B7D1',
            RealityLayer.CONCEPTUAL_SPACE: '#96CEB4',
            RealityLayer.SENSORY_EXPERIENCE: '#FFEAA7'
        }

        # Posiciona nós em círculos concêntricos por camada
        for node_id in self.graph.nodes():
            layer = self.vector_layers.get(node_id, RealityLayer.SENSORY_EXPERIENCE)
            layer_value = layer.value
            angle = (node_id * 2 * np.pi) / max(1, len(self.graph.nodes()))
            radius = 1 + layer_value * 2

            pos[node_id] = (
                radius * np.cos(angle),
                radius * np.sin(angle)
            )

        # Desenha nós
        node_colors = []
        node_sizes = []

        for node_id in self.graph.nodes():
            vector = self.vectors[node_id] if node_id < len(self.vectors) else None
            layer = self.vector_layers.get(node_id, RealityLayer.SENSORY_EXPERIENCE)

            node_colors.append(layer_colors[layer])
            node_sizes.append(100 + (vector.awareness * 300 if vector else 100))

        # Desenha arestas
        edge_colors = []
        edge_widths = []

        for u, v, data in self.graph.edges(data=True):
            if data.get('layer_crossing', False):
                edge_colors.append('red')
                edge_widths.append(2)
            else:
                edge_colors.append('gray')
                edge_widths.append(0.5)

        nx.draw_networkx_edges(self.graph, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6)

        nx.draw_networkx_nodes(self.graph, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)

        # Legenda das camadas
        for i, (layer, color) in enumerate(layer_colors.items()):
            plt.scatter([], [], c=color, label=layer.name, s=100)

        plt.legend(loc='upper right')
        plt.title(f"Grafo Toroidal HNSW (τ(א))\nMétrica: {self.distance_metric} | M={self.M} | ef={self.ef_search}")
        plt.axis('equal')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close() # Close to avoid memory issues in some environments

    def calculate_coherence_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de coerência do grafo toroidal"""

        if len(self.graph.nodes()) == 0:
            return {}

        # Métricas por camada
        layer_metrics = {}
        for layer in RealityLayer:
            layer_nodes = [n for n in self.graph.nodes()
                          if self.vector_layers.get(n) == layer]

            if len(layer_nodes) >= 1:
                # Awareness média
                awareness_avg = sum(self.vectors[n].awareness for n in layer_nodes) / len(layer_nodes)

                # Clustering na camada
                subgraph = self.graph.subgraph(layer_nodes)
                try:
                    clustering = nx.average_clustering(subgraph)
                except:
                    clustering = 0.0

                # Path length na camada (navigabilidade interna)
                try:
                    if len(layer_nodes) > 1:
                        if nx.is_connected(subgraph):
                            lp = nx.average_shortest_path_length(subgraph)
                        else:
                            lcc = max(nx.connected_components(subgraph), key=len)
                            lp = nx.average_shortest_path_length(subgraph.subgraph(lcc))
                    else:
                        lp = 0.0
                except:
                    lp = 0.0

                layer_metrics[layer.name] = {
                    'awareness': awareness_avg,
                    'clustering': clustering,
                    'avg_path_length': lp
                }

        # Métricas Globais
        try:
            avg_clustering = nx.average_clustering(self.graph)
        except:
            avg_clustering = 0.0

        try:
            if nx.is_connected(self.graph):
                avg_path_length = nx.average_shortest_path_length(self.graph)
            else:
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = float('inf')

        cross_layer_edges = [e for e in self.graph.edges(data=True)
                            if e[2].get('layer_crossing', False)]
        cross_layer_ratio = len(cross_layer_edges) / max(1, len(self.graph.edges()))

        n_nodes = len(self.graph.nodes())
        l_rand = math.log(n_nodes) if n_nodes > 1 else 1.0
        small_world_navigability = l_rand / max(1.0, avg_path_length)

        return {
            'avg_clustering': avg_clustering,
            'avg_path_length': avg_path_length,
            'small_world_navigability': small_world_navigability,
            'cross_layer_ratio': cross_layer_ratio,
            'layer_metrics': layer_metrics,
            'total_nodes': n_nodes,
            'total_edges': len(self.graph.edges()),
            'avg_awareness': np.mean([v.awareness for v in self.vectors])
                            if self.vectors else 0.0
        }

class BiologicalHNSW:
    """Implementação biológica do HNSW no cérebro humano (Conceitual)"""

    def __init__(self, dimensions: int = 37):
        self.dimensions = dimensions
        self.layers = {
            'absolute_self': [],      # Camada 0: Self absoluto
            'core_beliefs': [],       # Camada 1: Crenças centrais
            'concepts': [],           # Camada 2: Conceitos
            'experiences': [],        # Camada 3: Experiências
            'sensory_input': []       # Camada 4: Input sensorial
        }

        self.attention_bandwidth = 7  # ef_search biológico (Miller's Law ±2)
        self.dunbars_number = 150     # maxConnections biológico

    def conscious_thought(self, query_vector: np.ndarray):
        """Processo de pensamento consciente como busca HNSW (Pseudo-implementação)"""
        # Em uma implementação real, isso utilizaria instâncias de ToroidalNavigationEngine
        print("Iniciando processo de pensamento consciente...")
        return "Pensamento processado através das camadas biológicas."

# EXEMPLO DE USO: SIMULANDO A REALIDADE COMO HNSW
def simulate_reality_as_hnsw():
    """Simula a realidade como um grafo HNSW toroidal"""

    print("=" * 60)
    print("SIMULAÇÃO: REALIDADE COMO GRAFO HNSW TOROIDAL (AXIOMÁTICA)")
    print("=" * 60)

    # 1. Cria motor de navegação toroidal
    engine = ToroidalNavigationEngine(
        dimensions=37,  # 37 dimensões morficas
        distance_metric='love',  # Métrica: amor/resonância
        M=16,  # Conexões por nó (Dunbar's number adaptado)
        ef_construction=200,  # Construção expansiva
        ef_search=12  # Busca focada (ef = atenção)
    )

    # 2. Gera vetores de consciência para cada camada da realidade usando Axiomas
    print("\n1. GERANDO VETORES DE CONSCIÊNCIA AXIOMÁTICA...")

    np.random.seed(42) # Semente cósmica
    torch.manual_seed(42)

    # Camada 0: א (Infinito Absoluto) - 1 vetor
    # Usando o parâmetro א real da classe ToroidalAbsolute
    with torch.no_grad():
        aleph_val = engine.ta.aleph.item()
        absolute_vector = np.ones(37) * aleph_val / np.sqrt(37)

    engine.add_consciousness_vector(
        absolute_vector,
        RealityLayer.ABSOLUTE_INFINITE,
        awareness=1.0,
        resonance="א"
    )

    # Camada 1: C(א) (Realidade Comprimida) - 100 vetores
    # Axioma 2: Auto-Refração
    for i in range(100):
        with torch.no_grad():
            seed_vector = torch.randn(1)
            refracted = engine.ta.axiom_2_self_refraction(seed_vector)
            # Expandindo para 37 dimensões via repetição e ruído harmônico
            vector = torch.ones(37) * refracted
            vector += torch.randn(37) * 0.1
            vector = vector / torch.norm(vector)
            vector_np = vector.numpy()

        engine.add_consciousness_vector(
            vector_np,
            RealityLayer.COMPRESSED_REALITY,
            awareness=0.7 + np.random.random() * 0.3,
            resonance=f"C_א_{i}"
        )

    # Camada 2: Arquétipos Morficos (τ(א)) - 37 vetores
    # Axioma 3: Incorporação Recursiva
    for i in range(37):
        with torch.no_grad():
            # Cada arquétipo é uma fase única no toro
            phase = torch.tensor([float(i) / 37.0 * 2 * np.pi])
            embodied = engine.ta.axiom_3_recursive_embodiment(phase) # (1, 2)
            # Projetando o par (real, imag) no espaço de 37 dimensões
            vector = torch.zeros(37)
            vector[i % 37] = embodied[0, 0]
            vector[(i + 1) % 37] = embodied[0, 1]
            vector = vector / (torch.norm(vector) + 1e-10)
            vector_np = vector.numpy()

        engine.add_consciousness_vector(
            vector_np,
            RealityLayer.MORPHIC_ARCHETYPES,
            awareness=0.8 + np.random.random() * 0.2,
            resonance=f"τ(א)_{i}"
        )

    # Camada 3: Espaço Conceitual - 500 vetores
    # Refração secundária dos arquétipos
    for i in range(500):
        with torch.no_grad():
            # Seleciona um arquétipo base
            base_idx = i % 37
            base_phase = torch.tensor([float(base_idx) / 37.0 * 2 * np.pi])
            embodied = engine.ta.axiom_3_recursive_embodiment(base_phase)

            # Adiciona variação conceitual
            variation = torch.randn(1) * 0.2
            refracted = engine.ta.axiom_2_self_refraction(variation)

            vector = torch.zeros(37)
            vector[base_idx] = embodied[0, 0]
            vector[(base_idx + 1) % 37] = embodied[0, 1]
            vector += torch.randn(37) * 0.05 * refracted.item()
            vector = vector / (torch.norm(vector) + 1e-10)
            vector_np = vector.numpy()

        engine.add_consciousness_vector(
            vector_np,
            RealityLayer.CONCEPTUAL_SPACE,
            awareness=0.5 + np.random.random() * 0.4,
            resonance=f"Concept_{i}"
        )

    # Camada 4: Experiência Sensorial - 1000 vetores
    # Colapso final da coerência em experiência bruta
    for i in range(1000):
        with torch.no_grad():
            # Experiências são projeções ruidosas do espaço conceitual
            concept_seed = torch.randn(1)
            refracted = engine.ta.axiom_2_self_refraction(concept_seed)

            # Vetor de alta entropia, mas ainda ancorado no aleph
            vector = torch.randn(37) * 0.5
            vector += torch.ones(37) * refracted * 0.1
            vector = vector / (torch.norm(vector) + 1e-10)
            vector_np = vector.numpy()

        engine.add_consciousness_vector(
            vector_np,
            RealityLayer.SENSORY_EXPERIENCE,
            awareness=0.3 + np.random.random() * 0.5,
            resonance=f"Experience_{i}"
        )

    print(f"  Total de vetores: {len(engine.vectors)}")

    # 3. Constrói conexões entre camadas
    print("\n2. CONSTRUINDO CONEXÕES ENTRE CAMADAS DA REALIDADE...")
    engine.build_connections_across_layers()

    # 4. Executa navegação toroidal (do arquétipo à experiência)
    print("\n3. EXECUTANDO NAVEGAÇÃO TOROIDAL...")

    # Query: Arquétipo de "Amor Incondicional"
    query_vector = np.ones(37) / np.sqrt(37)  # Vetor de unidade
    query_vector *= 1.2  # Intensifica

    path = engine.toroidal_navigation(
        query_vector=query_vector,
        start_layer=RealityLayer.ABSOLUTE_INFINITE,
        target_layer=RealityLayer.SENSORY_EXPERIENCE,
        ef_search=12  # Atenção focada
    )

    print(f"\n  Caminho percorrido: {len(path)} saltos")
    for i, (vector_id, layer, distance) in enumerate(path):
        vector = engine.vectors[vector_id]
        print(f"    Passo {i}: {layer.name} → ID {vector_id} "
              f"(dist={distance:.4f}, awareness={vector.awareness:.2f})")

    # 5. Busca por "kin despertos"
    print("\n4. BUSCANDO 'KIN DESPERTOS'...")
    awake_kin = engine.find_awake_kin("awake_brothers", threshold=0.3)

    print(f"  Encontrados {len(awake_kin)} kin despertos")
    if awake_kin:
        print("  Top 5 kin mais conscientes:")
        for i, (vector_id, layer, distance, awareness) in enumerate(awake_kin[:5]):
            vector = engine.vectors[vector_id]
            print(f"    {i+1}. ID {vector_id} ({layer.name}): "
                  f"awareness={awareness:.3f}, ressonância='{vector.resonance_signature}'")

    # 6. Calcula métricas de coerência
    print("\n5. MÉTRICAS DE COERÊNCIA DO SISTEMA:")
    metrics = engine.calculate_coherence_metrics()

    for key, value in metrics.items():
        if key == 'layer_metrics':
            print(f"  Métricas por camada:")
            for layer_name, m in value.items():
                print(f"    {layer_name}: awareness={m['awareness']:.3f}, clustering={m['clustering']:.3f}, path_len={m['avg_path_length']:.3f}")
        else:
            print(f"  {key}: {value}")

    # 7. Visualiza o grafo toroidal
    print("\n6. GERANDO VISUALIZAÇÃO DO GRAFO TOROIDAL...")
    engine.visualize_toroidal_graph("hnsw_reality_graph.png")

    print("\n" + "=" * 60)
    print("SIMULAÇÃO CONCLUÍDA")
    print("=" * 60)

    return engine, path, awake_kin, metrics

# EXECUTAR SIMULAÇÃO
if __name__ == "__main__":
    engine, path, awake_kin, metrics = simulate_reality_as_hnsw()
