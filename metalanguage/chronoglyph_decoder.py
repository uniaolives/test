# chronoglyph_decoder.py
from metalanguage.chronoglyph_runtime import ChronoGraph, ChronoNode, QuantumState
from typing import Optional, Dict, List, Any

class BitSequenceDecoder:
    """
    Decodifica sequência de 85 bits como programa Chronoglyph
    Cada bit é interpretado como elemento de construção topológica
    """

    # Mapeamento de padrões de bits para elementos visuais
    PATTERNS = {
        '0000': ('void', 1.0),      # ● - ponto/vazio
        '0001': ('core', 0.2),      # ◉ pequeno
        '0010': ('core', 0.5),      # ◉ médio
        '0011': ('core', 1.0),      # ◉ grande (valor máximo)
        '0100': ('orbit', 0.5),     # ○ pequena
        '0101': ('orbit', 1.0),     # ○ grande
        '0110': ('state', 0.5),     # ◐
        '0111': ('state', 1.0),     # ◐ intenso
        '1000': ('possible', 0.5),  # ◑
        '1001': ('possible', 1.0),  # ◑ definido
        '1010': ('reference', 0.5), # ◎
        '1011': ('reference', 1.0), # ◎ forte
        '1100': ('fusion', 0.5),    # operador ⊕
        '1101': ('fusion', 1.0),    # ⊕ intenso
        '1110': ('flow', 0.5),      # conexão circular
        '1111': ('flow', 1.0),      # fluxo máximo
    }

    def __init__(self, bit_sequence: str):
        # Normaliza sequência (remove espaços, converte para string binária)
        self.bits = bit_sequence.replace(',', '').replace(' ', '').strip()
        self.length = len(self.bits)
        self.graph = ChronoGraph()

    def decode_to_chronograph(self) -> ChronoGraph:
        """
        Decodifica bits em grafo Chronoglyph executável
        """
        # Divide em blocos de 4 bits (nibbles) = elementos atômicos
        nibbles = [self.bits[i:i+4] for i in range(0, len(self.bits), 4)]

        # Estrutura: 42 + 42 + 1 bits = 10+10 nibbles + 1 bit
        # Interpreta como dois sistemas concêntricos + bit de controle

        system1_nibbles = nibbles[:10]  # Bloco 1: 40 bits
        system2_nibbles = nibbles[10:20]  # Bloco 2: 40 bits
        control_bit = self.bits[41] if len(self.bits) > 41 else '0'

        # Constrói sistema 1 (núcleo + órbitas)
        self._build_system('S1', system1_nibbles, control_bit)

        # Constrói sistema 2 (contexto/modificador)
        self._build_system('S2', system2_nibbles, control_bit)

        # Conecta sistemas (fusão se bit de controle = 1)
        if control_bit == '1':
            self._create_fusion('S1', 'S2')

        return self.graph

    def _build_system(self, sys_id: str, nibbles: List[str], control: str):
        """Constrói um sistema concêntrico a partir de nibbles"""
        if not nibbles:
            return

        # Núcleo: primeiro nibble significativo
        nucleus_value = 0
        for i, nib in enumerate(nibbles[:4]):
            if nib in self.PATTERNS:
                kind, intensity = self.PATTERNS[nib]
                if kind == 'core':
                    nucleus_value = int(nib, 2) * intensity

        nucleus_id = f"{sys_id}_nucleus"
        self.graph.add_node(ChronoNode(
            nucleus_id, 'core', nucleus_value,
            quantum_state=QuantumState.COLLAPSED if control == '1' else QuantumState.SUPERPOSITION
        ))

        # Órbitas: nibbles restantes
        for i, nib in enumerate(nibbles[4:], start=1):
            if nib not in self.PATTERNS:
                continue

            kind, intensity = self.PATTERNS[nib]
            orbit_id = f"{sys_id}_orbit{i}"

            # Mapeia tipos de padrão para tipos Chronoglyph
            chrono_kind = {
                'orbit': 'orbit',
                'state': 'state',
                'possible': 'possible',
                'reference': 'reference',
                'fusion': 'orbit',  # operador em órbita
                'flow': 'orbit',
                'void': 'void',
                'core': 'orbit',  # core em órbita = modificador
            }.get(kind, 'unknown')

            # Define estado quântico inicial por tipo
            q_state = QuantumState.COLLAPSED
            if kind in ['state', 'possible']:
                q_state = QuantumState.SUPERPOSITION
            elif kind == 'reference':
                q_state = QuantumState.ENTANGLED

            self.graph.add_node(ChronoNode(
                orbit_id, chrono_kind,
                value=int(nib, 2) * intensity,
                context={'orbit_order': i, 'intensity': intensity},
                quantum_state=q_state
            ))

            # Conexão radial
            self.graph.add_connection(nucleus_id, orbit_id, 'context')

            # Conexão circular com órbita anterior
            if i > 1:
                prev_orbit = f"{sys_id}_orbit{i-1}"
                self.graph.add_connection(prev_orbit, orbit_id, 'flows_to')

    def _create_fusion(self, sys1: str, sys2: str):
        """Cria operador de fusão entre dois sistemas"""
        # Busca núcleos
        n1 = f"{sys1}_nucleus"
        n2 = f"{sys2}_nucleus"

        if n1 in self.graph.nodes and n2 in self.graph.nodes:
            # Cria nó de fusão
            fusion_id = f"fusion_{sys1}_{sys2}"
            self.graph.add_node(ChronoNode(
                fusion_id, 'state',
                value=(self.graph.nodes[n1].value, self.graph.nodes[n2].value),
                quantum_state=QuantumState.SUPERPOSITION
            ))

            # Conecta sistemas à fusão
            self.graph.add_connection(n1, fusion_id, 'contributes')
            self.graph.add_connection(n2, fusion_id, 'contributes')

    def execute(self, input_values: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Executa o programa decodificado"""
        universes = self.graph.collapse(input_values)
        results = []
        for u in universes:
            result = self.graph.extract_result(u, 'nucleus')
            if result is not None:
                results.append({
                    'value': result,
                    'probability': u.probability,
                    'history': u.history
                })
        return results

    def to_svg(self) -> str:
        """Exporta programa como SVG visualizável"""
        # Gera SVG a partir do grafo
        svg_parts = ['<svg width="500" height="500" xmlns="http://www.w3.org/2000/svg">']

        # Sistemas concêntricos
        y_offset = 100
        for sys_id in ['S1', 'S2']:
            center_x = 150 if sys_id == 'S1' else 350
            center_y = y_offset

            # Núcleo
            nucleus = f"{sys_id}_nucleus"
            if nucleus in self.graph.nodes:
                node = self.graph.nodes[nucleus]
                r = 20 + (node.value if node.value else 0) * 2
                fill = 'black' if node.quantum_state.name == 'COLLAPSED' else 'gray'
                svg_parts.append(
                    f'<circle cx="{center_x}" cy="{center_y}" r="{r}" '
                    f'fill="{fill}" data-value="{node.value}"/>'
                )

            # Órbitas
            for i in range(1, 10):
                orbit = f"{sys_id}_orbit{i}"
                if orbit in self.graph.nodes:
                    node = self.graph.nodes[orbit]
                    r = 40 + i * 15
                    stroke = {
                        'orbit': 'black',
                        'state': 'blue',
                        'possible': 'green',
                        'reference': 'red'
                    }.get(node.kind, 'yellow')

                    dash = ' stroke-dasharray="5,5"' if node.kind == "possible" else ""
                    svg_parts.append(
                        f'<circle cx="{center_x}" cy="{center_y}" r="{r}" '
                        f'fill="none" stroke="{stroke}" stroke-width="2"{dash}/>'
                    )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

# Demonstração completa
def demo_85bit_chronoglyph():
    """Decodifica e executa a sequência de 85 bits"""

    # A sequência que apareceu em múltiplos contextos
    SEQUENCE_85 = "00001010111011000111110011010010000101011101100011111001101001000010101110"

    print("=" * 60)
    print("DECODIFICADOR CHRONOGLYPH: Sequência de 85 Bits")
    print("=" * 60)

    decoder = BitSequenceDecoder(SEQUENCE_85)

    print(f"\nSequência de entrada: {SEQUENCE_85[:42]}...{SEQUENCE_85[42:]}")
    print(f"Comprimento: {len(SEQUENCE_85)} bits")
    print(f"Bit de controle (posição 42): {SEQUENCE_85[41]}")

    # Decodifica
    cg = decoder.decode_to_chronograph()

    print(f"\nGrafo decodificado:")
    print(f"  Nós: {len(cg.nodes)}")
    print(f"  Arestas: {cg.graph.number_of_edges()}")
    print(f"  Tipos: {set(n.kind for n in cg.nodes.values())}")

    # Executa
    print(f"\nExecução (colapso):")
    results = decoder.execute()

    print(f"  Universos gerados: {len(results)}")
    for i, r in enumerate(results[:3]):
        print(f"    Resultado {i}: valor={r['value']}, P={r['probability']:.3f}")

    # Gera visualização
    svg_output = decoder.to_svg()
    with open("85bit_program.svg", "w") as f:
        f.write(svg_output)
    print(f"\nVisualização salva em: 85bit_program.svg")

    # Interpretação astroquímica
    print(f"\nInterpretação como molécula:")
    bit_42 = SEQUENCE_85[41]
    if bit_42 == '0':
        print("  Bit 42 = 0: Fase abiótica (H₂O-like)")
        print("  Predição: Molécula simples, equilíbrio termodinâmico")
    else:
        print("  Bit 42 = 1: Fase informacional (ureia-like)")
        print("  Predição: Molécula complexa, ligação peptídica, desvio do equilíbrio")

    return decoder, results

if __name__ == "__main__":
    demo_85bit_chronoglyph()
