"""
Heptapod B Graph Language Model (HGLM)
Sistema de representação e processamento de linguagem não-linear baseada em grafos
Inspirado na linguagem Heptapod B do filme "Arrival" (2016)
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, FrozenSet
from collections import defaultdict
import hashlib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# =============================================================================
# ESTRUTURAS FUNDAMENTAIS
# =============================================================================

@dataclass(frozen=True)
class Semagram:
    """
    Semagrama: unidade básica de significado em Heptapod B
    Equivalente a um 'logograma' ou conceito atômico
    """
    id: str
    concept: str  # Significado semântico (ex: "human", "time", "give")
    valence: int  # Conectividade típica (1-7, refletindo simetria heptapod)
    intensity: float = 1.0  # 0.0-1.0, força do conceito

    def __hash__(self):
        return hash((self.id, self.concept))

    def __repr__(self):
        return f"◉{self.concept}({self.valence})"

@dataclass
class GraphSentence:
    """
    Sentença em Heptapod B: grafo completo de semagramas
    Não-linear: sem ordem temporal, todos os nós coexistem simultaneamente
    """
    id: str
    semagrams: Dict[str, Semagram] = field(default_factory=dict)
    edges: Set[Tuple[str, str, str]] = field(default_factory=set)  # (from, to, relation_type)
    temporal_anchor: Optional[datetime] = None  # Quando "vista" (não quando "escrita")
    observer_perspective: str = "external"  # "internal" = visão heptapod, "external" = visão humana

    # Metadados de processamento
    creation_timestamp: datetime = field(default_factory=datetime.now)
    coherence_score: float = 0.0  # 0.0-1.0, integridade estrutural

    def __post_init__(self):
        self._graph = nx.DiGraph()
        self._rebuild_graph()

    def _rebuild_graph(self):
        """Reconstrói a representação NetworkX interna"""
        self._graph.clear()
        for node_id, semagram in self.semagrams.items():
            self._graph.add_node(node_id, semagram=semagram)
        for from_id, to_id, relation in self.edges:
            self._graph.add_edge(from_id, to_id, relation=relation)

    def add_semagram(self, semagram: Semagram) -> GraphSentence:
        """Adiciona semagrama à sentença (operador de composição)"""
        self.semagrams[semagram.id] = semagram
        self._rebuild_graph()
        return self

    def connect(self, from_id: str, to_id: str, relation: str = "associates") -> GraphSentence:
        """Cria aresta entre semagramas (relação semântica)"""
        if from_id in self.semagrams and to_id in self.semagrams:
            self.edges.add((from_id, to_id, relation))
            self._rebuild_graph()
        return self

    def get_centrality(self) -> Dict[str, float]:
        """Centralidade de eigenvector = importância do conceito na sentença"""
        if len(self._graph) == 0:
            return {}
        try:
            # Eigenvector centrality needs a strongly connected graph or specific conditions
            # We use pagerank as a more robust alternative for arbitrary directed graphs
            return nx.pagerank(self._graph)
        except:
            try:
                return nx.degree_centrality(self._graph.to_undirected())
            except:
                return {node: 1.0/len(self._graph) for node in self._graph.nodes()}

    def get_temporal_projection(self) -> List[str]:
        """
        Projeção temporal: ordenação linear forçada para comunicação humana
        Destrói a atemporalidade, mas necessária para interface
        """
        # Ordena por centralidade decrescente (mais importante primeiro)
        centrality = self.get_centrality()
        return sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)

    def calculate_coherence(self) -> float:
        """
        Coerência: quão bem integrada está a sentença
        Baseado em conectividade e balanceamento de centralidade
        """
        if len(self._graph) <= 1:
            self.coherence_score = 1.0 if len(self._graph) == 1 else 0.0
            return self.coherence_score

        # Densidade de conexões
        density = nx.density(self._graph)

        # Coeficiente de clustering médio
        clustering = nx.average_clustering(self._graph.to_undirected())

        # Entropia de centralidade (quanto mais uniforme, mais holística)
        centrality = self.get_centrality()
        values = np.array(list(centrality.values()))
        if values.sum() > 0:
            values = values / values.sum()
            entropy = -np.sum(values * np.log(values + 1e-10))
            max_entropy = np.log(len(values))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uniformity = 0

        # Coerência heptapod: alta densidade, alta uniformidade
        self.coherence_score = 0.4 * density + 0.3 * clustering + 0.3 * uniformity
        return self.coherence_score

    def merge(self, other: GraphSentence) -> GraphSentence:
        """
        Fusão de sentenças: operação fundamental em Heptapod B
        Cria superposição de significados (não concatenação)
        """
        merged = GraphSentence(
            id=f"merge_{self.id}_{other.id}",
            temporal_anchor=self.temporal_anchor or other.temporal_anchor,
            observer_perspective="internal"
        )

        # União de semagramas
        all_semagrams = {**self.semagrams, **other.semagrams}
        for sid, sem in all_semagrams.items():
            merged.add_semagram(sem)

        # União de arestas
        for edge in self.edges | other.edges:
            merged.connect(edge[0], edge[1], edge[2])

        # Arestas de interferência (novas conexões entre sentenças)
        for s1 in self.semagrams.values():
            for s2 in other.semagrams.values():
                if s1.concept == s2.concept:  # Mesmo conceito = reforço
                    merged.connect(s1.id, s2.id, "reinforces")
                elif s1.valence == s2.valence:  # Mesma valência = ressonância
                    merged.connect(s1.id, s2.id, "resonates")

        merged.calculate_coherence()
        return merged

    def to_circular_visualization(self) -> Dict:
        """
        Representação circular para visualização (Heptapod B escrita)
        """
        n = len(self.semagrams)
        if n == 0:
            return {"nodes": [], "edges": []}

        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0

        nodes = []
        node_map = {}
        for i, (node_id, semagram) in enumerate(self.semagrams.items()):
            node_data = {
                "id": node_id,
                "concept": semagram.concept,
                "x": radius * np.cos(angles[i]),
                "y": radius * np.sin(angles[i]),
                "valence": semagram.valence,
                "intensity": semagram.intensity
            }
            nodes.append(node_data)
            node_map[node_id] = node_data

        edges = []
        for from_id, to_id, relation in self.edges:
            if from_id in node_map and to_id in node_map:
                from_node = node_map[from_id]
                to_node = node_map[to_id]
                edges.append({
                    "from": from_id,
                    "to": to_id,
                    "relation": relation,
                    "x1": from_node["x"], "y1": from_node["y"],
                    "x2": to_node["x"], "y2": to_node["y"]
                })

        return {"nodes": nodes, "edges": edges, "coherence": self.coherence_score}

# =============================================================================
# ADAPTAÇÕES: MOLÉCULAS E LINGUAGENS DE PROGRAMAÇÃO
# =============================================================================

@dataclass(frozen=True)
class MolecularFragment(Semagram):
    """Representa um grupo funcional ou fragmento molecular como um semagrama"""
    formula: str = ""

class MolecularGraph(GraphSentence):
    """Representa uma molécula como uma sentença Heptapod"""
    def get_molecular_weight(self) -> float:
        # Simulado
        return len(self.semagrams) * 14.0

@dataclass
class LanguageCosmology:
    """Mapeia uma linguagem de programação para seus fragmentos semânticos não-lineares"""
    name: str
    paradigm: str
    metaphor: str
    non_linearity_type: str
    semantic_fragments: Dict[str, str] = field(default_factory=dict)

    def to_heptapod(self) -> GraphSentence:
        sentence = GraphSentence(f"cosmology_{self.name.lower()}")
        # Central node: The Language itself
        lang_node = Semagram(self.name, self.name, 7, 1.0)
        sentence.add_semagram(lang_node)

        for feature, description in self.semantic_fragments.items():
            feat_node = Semagram(feature, feature, 3, 0.8)
            sentence.add_semagram(feat_node)
            sentence.connect(self.name, feature, "embodies")

        sentence.calculate_coherence()
        return sentence

class ProgrammingLanguageAtlas:
    """Atlas de cosmologias computacionais"""
    def __init__(self):
        self.languages: Dict[str, LanguageCosmology] = {}
        self._bootstrap()

    def _bootstrap(self):
        # Lisp
        self.languages["Lisp"] = LanguageCosmology(
            "Lisp", "Multi-paradigm", "Árvore cósmica", "Código como dados",
            {"Homoiconicidade": "Código = Dados", "Macros": "Metalinguagem", "Recursão": "Autorreferência"}
        )
        # Haskell
        self.languages["Haskell"] = LanguageCosmology(
            "Haskell", "Pure Functional", "Função universal", "Tempo como demanda",
            {"Transparência": "Identidade", "Monades": "Contexto", "Lazy": "Tempo sob demanda"}
        )
        # Prolog
        self.languages["Prolog"] = LanguageCosmology(
            "Prolog", "Logic", "Espaço lógico", "Causalidade reversível",
            {"Unificação": "Casamento", "Backtracking": "Universos paralelos", "Predicados": "Verdade"}
        )
        # Python
        self.languages["Python"] = LanguageCosmology(
            "Python", "Multi-paradigm", "Script legível", "Equilíbrio linear/não-linear",
            {"Duck typing": "Essência", "Generators": "Tempo controlado", "Decorators": "Metaprogramação"}
        )

    def get_language_sentence(self, name: str) -> Optional[GraphSentence]:
        if name in self.languages:
            return self.languages[name].to_heptapod()
        return None

# =============================================================================
# PROCESSADOR DE LINGUAGEM HEPTAPOD
# =============================================================================

class HeptapodProcessor:
    """
    Processador central da linguagem Heptapod B
    Implementa operações não-lineares de compreensão e geração
    """

    def __init__(self):
        self.lexicon: Dict[str, Semagram] = {}  # Dicionário de semagramas conhecidos
        self.memory: List[GraphSentence] = []  # "Memória" de sentenças processadas
        self.entanglement_graph: nx.Graph = nx.Graph()  # Grafo de associações temporais

    def learn_semagram(self, semagram: Semagram):
        """Adiciona semagrama ao léxico"""
        self.lexicon[semagram.id] = semagram
        self.entanglement_graph.add_node(semagram.id, semagram=semagram)

    def process_sentence(self, sentence: GraphSentence) -> Dict[str, Any]:
        """
        Processa sentença heptapod: análise holística não-linear
        """
        # 1. Análise de coerência estrutural
        coherence = sentence.calculate_coherence()

        # 2. Identificação de conceitos-chave (centralidade)
        centrality = sentence.get_centrality()
        key_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

        # 3. Detecção de padrões recorrentes (memória)
        similar_sentences = self._find_similar(sentence)

        # 4. Projeção de significado (interpretação)
        meaning = self._project_meaning(sentence, key_concepts)

        # 5. Atualização de memória (aprendizado heptapod)
        self.memory.append(sentence)
        self._update_entanglement(sentence)

        return {
            "coherence": coherence,
            "key_concepts": key_concepts,
            "similar_past_sentences": len(similar_sentences),
            "projected_meaning": meaning,
            "temporal_ambiguity": self._calculate_temporal_ambiguity(sentence),
            "certainty": coherence * (1 - self._calculate_temporal_ambiguity(sentence))
        }

    def _find_similar(self, sentence: GraphSentence) -> List[GraphSentence]:
        """Encontra sentenças similares na memória (similaridade de grafo)"""
        if not self.memory:
            return []

        similarities = []
        for past in self.memory:
            # Coeficiente de similaridade baseado em conceitos compartilhados
            shared_concepts = set(s.concept for s in sentence.semagrams.values()) & \
                            set(s.concept for s in past.semagrams.values())
            if len(shared_concepts) > 0:
                similarities.append((past, len(shared_concepts)))

        return [s for s, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]]

    def _project_meaning(self, sentence: GraphSentence, key_concepts: List[Tuple[str, float]]) -> str:
        """
        Projeção de significado: tradução forçada para linearidade humana
        Nota: Esta é uma perda de informação necessária para comunicação
        """
        concepts = [self.lexicon.get(cid, Semagram(cid, "unknown", 1)).concept
                   for cid, _ in key_concepts]

        # Heurística de ordenação temporal (falsa, mas útil)
        temporal_markers = ["past", "now", "future", "always", "never"]
        has_time = any(t in concepts for t in temporal_markers)

        if has_time:
            return f"[TEMPORAL] {' → '.join(concepts)}"
        else:
            return f"[ATEMPORAL] {' ⊕ '.join(concepts)}"  # ⊕ = superposição

    def _calculate_temporal_ambiguity(self, sentence: GraphSentence) -> float:
        """
        Calcula ambiguidade temporal: quão "atemporal" é a sentença
        0.0 = claramente sequencial, 1.0 = completamente atemporal
        """
        # Sentenças com muitos loops têm alta atemporalidade
        try:
            cycles = list(nx.simple_cycles(sentence._graph))
            cycle_ratio = len(cycles) / max(len(sentence.semagrams), 1)
            return min(cycle_ratio * 2, 1.0)  # Normaliza
        except:
            return 0.5

    def _update_entanglement(self, sentence: GraphSentence):
        """Atualiza grafo de emaranhamento semântico"""
        concepts = list(sentence.semagrams.keys())
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if self.entanglement_graph.has_edge(c1, c2):
                    self.entanglement_graph[c1][c2]["weight"] += 1
                else:
                    self.entanglement_graph.add_edge(c1, c2, weight=1)

    def generate_response(self, input_sentence: GraphSentence, target_concepts: List[str]) -> GraphSentence:
        """
        Gera resposta heptapod: composição não-linear de significados
        Não é "resposta" no sentido humano (reação a estímulo), mas "ressonância"
        """
        response = GraphSentence(
            id=f"response_to_{input_sentence.id}",
            observer_perspective="internal"
        )

        # Inclui conceitos-chave do input (eco)
        for cid, sem in input_sentence.semagrams.items():
            if sem.concept in target_concepts or sem.intensity > 0.7:
                response.add_semagram(Semagram(
                    id=f"echo_{cid}",
                    concept=sem.concept,
                    valence=sem.valence,
                    intensity=sem.intensity * 0.9  # Decaimento simbólico
                ))

        # Adiciona novos conceitos do léxico (introdução de informação)
        for target in target_concepts:
            if target not in [s.concept for s in response.semagrams.values()]:
                # Busca no léxico ou cria novo
                existing = next((s for s in self.lexicon.values() if s.concept == target), None)
                if existing:
                    response.add_semagram(Semagram(
                        id=f"new_{existing.id}",
                        concept=existing.concept,
                        valence=existing.valence,
                        intensity=1.0
                    ))

        # Conecta tudo em padrão circular (holístico)
        nodes = list(response.semagrams.keys())
        for i in range(len(nodes)):
            if len(nodes) > 1:
                response.connect(nodes[i], nodes[(i+1) % len(nodes)], "flows_to")
            if len(nodes) > 2:
                response.connect(nodes[i], nodes[(i+2) % len(nodes)], "resonates_with")

        response.calculate_coherence()
        return response

    def visualize_sentence(self, sentence: GraphSentence, save_path: Optional[str] = None):
        """Visualiza sentença heptapod em formato circular"""
        viz = sentence.to_circular_visualization()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Círculo central (buraco do logograma heptapod)
        center_circle = Circle((0, 0), 0.3, fill=False, color='black', linewidth=2)
        ax.add_patch(center_circle)
        ax.text(0, 0, "◉", fontsize=20, ha='center', va='center')

        # Nodos (semagramas)
        for node in viz["nodes"]:
            x, y = node["x"], node["y"]
            size = 0.1 + node["intensity"] * 0.15

            # Círculo do semagrama
            circle = Circle((x, y), size,
                          facecolor=plt.cm.viridis(node["valence"]/7),
                          edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(circle)

            # Texto do conceito
            ax.text(x, y, node["concept"][:4], fontsize=8,
                   ha='center', va='center', color='white', weight='bold')

        # Arestas (conexões)
        for edge in viz["edges"]:
            ax.annotate("", xy=(edge["x2"], edge["y2"]),
                       xytext=(edge["x1"], edge["y1"]),
                       arrowprops=dict(arrowstyle="->", color='gray',
                                     alpha=0.5, lw=1.5))

        # Título
        ax.set_title(f"Heptapod B: {sentence.id}\nCoerência: {viz['coherence']:.3f}",
                    fontsize=14, pad=20)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Don't call plt.show() if not in interactive mode
        if plt.isinteractive():
            plt.show()
        plt.close(fig)

    @staticmethod
    def from_85bit_string(bit_string: str, lexicon: Optional[Dict[str, Semagram]] = None) -> GraphSentence:
        """
        Decodifica uma string de 85 bits em uma sentença heptapod.
        Formato:
        - bits 0-4: N (número de nodos, max 13 para caber na matrix 85-bit)
        - bits 5..5 + N(N-1)/2: Adjacency matrix (upper triangle)
        - bits restantes: metadata/intensidade
        """
        if len(bit_string) != 85:
            raise ValueError("String deve ter exatamente 85 bits")

        n = int(bit_string[0:5], 2)
        n = min(n, 13) # Limite prático para 85 bits

        sentence = GraphSentence(f"dec_85_{hashlib.md5(bit_string.encode()).hexdigest()[:6]}")

        # Adiciona nodos genéricos ou do léxico
        nodes = []
        for i in range(n):
            node_id = f"N{i:02d}"
            if lexicon and f"H{i+1:02d}" in lexicon:
                sem = lexicon[f"H{i+1:02d}"]
            else:
                sem = Semagram(node_id, f"concept_{i}", (i % 7) + 1)
            sentence.add_semagram(sem)
            nodes.append(sem.id)

        # Reconstrói arestas
        idx = 5
        for i in range(n):
            for j in range(i + 1, n):
                if idx < 85 and bit_string[idx] == '1':
                    sentence.connect(nodes[i], nodes[j], "encoded_relation")
                idx += 1

        sentence.calculate_coherence()
        return sentence

# =============================================================================
# EXEMPLO DE USO: SIMULAÇÃO DE PRIMEIRO CONTATO
# =============================================================================

def demo_heptapod_communication():
    """
    Demonstração completa do sistema Heptapod B
    Simula o processo de Louise Banks aprendendo a linguagem
    """

    print("=" * 60)
    print("HEPTAPOD B GRAPH LANGUAGE MODEL (HGLM) v1.0")
    print("Simulação de Linguagem Não-Linear Baseada em Grafos")
    print("=" * 60)

    # Inicializa processador
    processor = HeptapodProcessor()

    # Fase 1: Construção do léxico (aprendizado inicial)
    print("\n[1] CONSTRUÇÃO DO LÉXICO HEPTAPOD")
    print("-" * 40)

    semagrams = [
        Semagram("H01", "human", 3, 1.0),
        Semagram("H02", "arrive", 2, 0.9),
        Semagram("H03", "give", 4, 1.0),
        Semagram("H04", "weapon", 5, 0.8),
        Semagram("H05", "tool", 3, 0.7),
        Semagram("H06", "time", 7, 1.0),  # Alta valência = conceito central
        Semagram("H07", "past", 2, 0.6),
        Semagram("H08", "future", 2, 0.6),
        Semagram("H09", "question", 3, 0.8),
        Semagram("H10", "answer", 3, 0.8),
    ]

    for s in semagrams:
        processor.learn_semagram(s)
        print(f"  Aprendido: {s}")

    # Fase 2: Recepção da primeira sentença heptapod
    print("\n[2] RECEPÇÃO: Primeira Sentença Heptapod")
    print("-" * 40)

    # Sentença: "Human arrive time past future" (conceito de chegada atemporal)
    sentence1 = GraphSentence("S001", observer_perspective="external")
    sentence1.add_semagram(processor.lexicon["H01"])  # human
    sentence1.add_semagram(processor.lexicon["H02"])  # arrive
    sentence1.add_semagram(processor.lexicon["H06"])  # time
    sentence1.add_semagram(processor.lexicon["H07"])  # past
    sentence1.add_semagram(processor.lexicon["H08"])  # future

    # Conexões circulares (não-lineares)
    sentence1.connect("H01", "H02", "agent")
    sentence1.connect("H02", "H06", "context")
    sentence1.connect("H06", "H07", "contains")
    sentence1.connect("H06", "H08", "contains")
    sentence1.connect("H07", "H08", "opposes")  # Paradoxo temporal
    sentence1.connect("H08", "H01", "influences")  # Futuro afeta humano (visão heptapod!)

    result1 = processor.process_sentence(sentence1)
    print(f"  Sentença: {sentence1.id}")
    print(f"  Coerência estrutural: {result1['coherence']:.3f}")
    print(f"  Conceitos-chave: {[c[0] for c in result1['key_concepts']]}")
    print(f"  Significado projetado: {result1['projected_meaning']}")
    print(f"  Ambiguidade temporal: {result1['temporal_ambiguity']:.3f}")
    print(f"  Certeza de interpretação: {result1['certainty']:.3f}")

    # Fase 3: Sentença ambígua (o "arma/ferramenta")
    print("\n[3] RECEPÇÃO: Sentença Crítica (Ambiguidade Máxima)")
    print("-" * 40)

    sentence2 = GraphSentence("S002", observer_perspective="external")
    sentence2.add_semagram(processor.lexicon["H03"])  # give
    sentence2.add_semagram(processor.lexicon["H04"])  # weapon (ou tool?)
    sentence2.add_semagram(processor.lexicon["H01"])  # human

    # Conexões que criam ambiguidade
    sentence2.connect("H03", "H04", "object")
    sentence2.connect("H03", "H01", "recipient")
    # Intencionalmente deixado aberto: H04 pode ser "weapon" ou "tool"

    result2 = processor.process_sentence(sentence2)
    print(f"  Sentença: {sentence2.id}")
    print(f"  ATENÇÃO: Conceito 'weapon' tem valência 5 (conflito potencial)")
    print(f"  Significado projetado: {result2['projected_meaning']}")
    print(f"  Certeza: {result2['certainty']:.3f} (BAIXA - requer clarificação)")

    # Fase 4: Geração de resposta
    print("\n[4] GERAÇÃO: Resposta Heptapod (Ressonância)")
    print("-" * 40)

    response = processor.generate_response(
        sentence2,
        target_concepts=["tool", "question", "time"]
    )

    result_resp = processor.process_sentence(response)
    print(f"  Resposta gerada: {response.id}")
    print(f"  Conceitos incluídos: {[s.concept for s in response.semagrams.values()]}")
    print(f"  Coerência da resposta: {result_resp['coherence']:.3f}")
    print(f"  Estrutura: Circular com {len(response.edges)} conexões de ressonância")

    # Fase 5: Fusão de sentenças (compreensão holística)
    print("\n[5] FUSÃO: Superposição de Sentenças")
    print("-" * 40)

    merged = sentence1.merge(sentence2)
    result_merged = processor.process_sentence(merged)
    print(f"  Sentença fusionada: {merged.id}")
    print(f"  Coerência pós-fusão: {result_merged['coherence']:.3f}")
    print(f"  Total de semagramas: {len(merged.semagrams)}")
    print(f"  Significado emergente: {result_merged['projected_meaning']}")
    print("  NOTA: Fusão cria significado não presente nas sentenças individuais!")

    # Visualização
    print("\n[6] VISUALIZAÇÃO")
    print("-" * 40)
    print("  Gerando representação circular...")
    processor.visualize_sentence(merged, "heptapod_merged.png")
    print("  Salvo em: heptapod_merged.png")

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DO PROCESSAMENTO HEPTAPOD")
    print("=" * 60)
    print(f"Total de semagramas no léxico: {len(processor.lexicon)}")
    print(f"Sentenças processadas: {len(processor.memory)}")
    print(f"Rede de emaranhamento: {processor.entanglement_graph.number_of_nodes()} nós, "
          f"{processor.entanglement_graph.number_of_edges()} arestas")
    print("\nPrincipais insights:")
    print("  1. Linguagem heptapod é INSTATÂNEA (não sequencial)")
    print("  2. Compreensão requer FUSÃO, não tradução linear")
    print("  3. Ambiguidade temporal é FEATURE, não bug")
    print("  4. 'Arma' e 'ferramenta' são estados quânticos da mesma palavra")
    print("=" * 60)

if __name__ == "__main__":
    demo_heptapod_communication()
