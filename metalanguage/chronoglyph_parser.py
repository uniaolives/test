# chronoglyph_parser.py
import xml.etree.ElementTree as ET
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import networkx as nx

@dataclass
class GeometricSignature:
    """Assinatura geométrica de um elemento visual"""
    x: float
    y: float
    radius: float
    kind: str  # 'circle', 'ring', 'point', 'arc'
    attributes: Dict[str, str]

class SVGTopoParser:
    """
    Parser que extrai topologia de círculos de SVG
    Converte desenho visual em grafo semântico
    """

    # Mapeamento visual → semântico
    KIND_MAP = {
        'filled_circle': 'core',      # ◉
        'empty_circle': 'orbit',      # ○
        'dashed_circle': 'possible',  # ◑
        'half_circle': 'state',       # ◐
        'double_circle': 'reference', # ◎
        'small_point': 'void',        # ●
    }

    def __init__(self, svg_content: str):
        self.svg = ET.fromstring(svg_content)
        self.elements: List[GeometricSignature] = []
        self.graph = nx.MultiDiGraph()

    def extract_circles(self):
        """Extrai círculos e anéis do SVG"""
        # Namespace SVG
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        for circle in self.svg.findall('.//svg:circle', ns):
            cx = float(circle.get('cx', 0))
            cy = float(circle.get('cy', 0))
            r = float(circle.get('r', 0))
            style = circle.get('style', '')
            fill = circle.get('fill', 'none')

            # Classificação por preenchimento e estilo
            kind = self._classify_visual(fill, style, r)

            self.elements.append(GeometricSignature(
                x=cx, y=cy, radius=r, kind=kind,
                attributes=dict(circle.attrib)
            ))

        # Ordena por raio (para detectar órbitas concêntricas)
        self.elements.sort(key=lambda e: e.radius)

    def _classify_visual(self, fill: str, style: str, radius: float) -> str:
        """Classifica elemento visual em tipo semântico"""
        if radius < 3:  # Pequeno = ponto
            return 'small_point'
        if fill == 'none' or fill == 'transparent':
            if 'stroke-dasharray' in style:
                return 'dashed_circle'
            return 'empty_circle'
        if fill == 'url(#double)':  # Pattern especial
            return 'double_circle'
        if 'opacity: 0.5' in style:
            return 'half_circle'
        return 'filled_circle'

    def build_topology(self):
        """Constrói topologia de órbitas concêntricas"""
        # Agrupa por proximidade espacial (centros próximos = mesmo sistema)
        systems = self._cluster_by_proximity()

        for system_id, elements in systems.items():
            self._process_system(system_id, elements)

    def _cluster_by_proximity(self, threshold: float = 10.0) -> Dict[int, List[GeometricSignature]]:
        """Agrupa elementos em sistemas concêntricos"""
        if not self.elements:
            return {}

        systems = {}
        current_system = 0
        used = set()

        for i, elem in enumerate(self.elements):
            if i in used:
                continue

            # Novo sistema centrado neste elemento
            systems[current_system] = [elem]
            used.add(i)

            # Busca elementos próximos (mesmo centro aproximado)
            for j, other in enumerate(self.elements[i+1:], start=i+1):
                if j in used:
                    continue
                dist = np.sqrt((elem.x - other.x)**2 + (elem.y - other.y)**2)
                if dist < threshold:
                    systems[current_system].append(other)
                    used.add(j)

            current_system += 1

        return systems

    def _process_system(self, system_id: int, elements: List[GeometricSignature]):
        """Processa um sistema concêntrico"""
        # Ordena por raio (do menor para o maior)
        ordered = sorted(elements, key=lambda e: e.radius)

        # O menor é o núcleo (se for filled_circle)
        if ordered and ordered[0].kind == 'filled_circle':
            nucleus = ordered[0]
            nucleus_id = f"sys{system_id}_nucleus"
            self.graph.add_node(
                nucleus_id,
                kind='core',
                value=self._extract_value(nucleus),
                position=(nucleus.x, nucleus.y),
                radius=nucleus.radius
            )

            # Órbitas subsequentes
            for i, orbit in enumerate(ordered[1:], start=1):
                orbit_id = f"sys{system_id}_orbit{i}"
                self.graph.add_node(
                    orbit_id,
                    kind=self.KIND_MAP.get(orbit.kind, 'unknown'),
                    position=(orbit.x, orbit.y),
                    radius=orbit.radius
                )
                # Conexão radial: núcleo → órbita
                self.graph.add_edge(
                    nucleus_id, orbit_id,
                    relation='context',
                    order=i
                )
                # Conexão circular (se houver próxima órbita)
                if i < len(ordered) - 1:
                    next_orbit = ordered[i+1]
                    next_id = f"sys{system_id}_orbit{i+1}"
                    self.graph.add_edge(
                        orbit_id, next_id,
                        relation='flows_to',
                        type='circular'
                    )

    def _extract_value(self, element: GeometricSignature) -> Optional[Any]:
        """Extrai valor semântico de atributos SVG"""
        # Tenta extrair de data-value, ou inferir de posição/cor
        value_attr = element.attributes.get('data-value')
        if value_attr:
            try:
                return int(value_attr)
            except:
                try:
                    return float(value_attr)
                except:
                    return value_attr

        # Inferência por cor (ex: #FF0000 = 255, 0, 0)
        fill = element.attributes.get('fill', '')
        if fill.startswith('#'):
            r = int(fill[1:3], 16)
            g = int(fill[3:5], 16)
            b = int(fill[5:7], 16)
            return (r, g, b)  # Como tuple semântico

        return None

    def to_chronograph(self) -> Any:
        """Exporta para estrutura ChronoGraph executável"""
        from metalanguage.chronoglyph_runtime import ChronoGraph, ChronoNode

        cg = ChronoGraph()

        for node_id, data in self.graph.nodes(data=True):
            node = ChronoNode(
                node_id=node_id,
                kind=data.get('kind', 'unknown'),
                value=data.get('value'),
                context={
                    'position': data.get('position'),
                    'radius': data.get('radius'),
                    'orbit_order': data.get('order', 0)
                }
            )
            cg.add_node(node)

        for u, v, data in self.graph.edges(data=True):
            cg.add_connection(u, v, data.get('relation', 'associates'))

        return cg

    def visualize_parsing(self):
        """Visualiza o resultado do parsing para depuração"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Desenha elementos originais
        for elem in self.elements:
            color = {
                'filled_circle': 'black',
                'empty_circle': 'white',
                'dashed_circle': 'gray',
                'half_circle': 'blue',
                'double_circle': 'red',
                'small_point': 'green'
            }.get(elem.kind, 'yellow')

            if elem.kind == 'filled_circle':
                circle = plt.Circle((elem.x, elem.y), elem.radius,
                                  color=color, fill=True, alpha=0.6)
            else:
                circle = plt.Circle((elem.x, elem.y), elem.radius,
                                  color=color, fill=False, linewidth=2)
            ax.add_patch(circle)

        # Desenha grafo parseado
        pos = nx.get_node_attributes(self.graph, 'position')
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_size=50, alpha=0.3)
        nx.draw_networkx_edges(self.graph, pos, ax=ax, alpha=0.2, arrows=True)

        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # SVG coordenadas
        plt.title("Chronoglyph: Parsing Visual → Topológico")
        plt.show()

# Exemplo de SVG de entrada
EXAMPLE_SVG = '''<?xml version="1.0"?>
<svg width="500" height="500" xmlns="http://www.w3.org/2000/svg">
  <!-- Sistema 1: ◉5 ⊕ ○+3 -->
  <circle cx="100" cy="100" r="20" fill="#000000" data-value="5"/>
  <circle cx="100" cy="100" r="40" fill="none" stroke="black"
          stroke-width="2" data-operator="add" data-operand="3"/>

  <!-- Sistema 2: ◐estado -->
  <circle cx="300" cy="100" r="30" fill="blue" opacity="0.5"/>

  <!-- Ponto de fusão -->
  <circle cx="200" cy="200" r="3" fill="green"/>
</svg>'''
