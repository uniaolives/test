# scripts/ascii_runtime.py
# Renderizador em tempo real do hipergrafo Arkhe(N) via ASCII

import numpy as np
import time
import os
from typing import List, Tuple, Dict
from papercoder_kernel.quantum.safe_core import SafeCore

class ASCIIHypergraphRenderer:
    """
    Renderizador ASCII do hipergrafo Arkhe(N).
    Converte estados quânticos e métricas de coerência em arte ASCII.
    """

    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.frame = 0

        # Paleta de caracteres por nível de coerência
        self.coherence_chars = " .:-=+*#%@"
        # Paleta para Φ (informação integrada) – ANSI colors
        self.phi_colors = [f"\033[3{i}m" for i in range(1, 8)]  # ANSI 31-37

    def render_node(self, x: int, y: int, coherence: float, phi: float):
        """
        Renderiza um nó na posição (x,y) com base em sua coerência.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        idx = int(coherence * (len(self.coherence_chars) - 1))
        char = self.coherence_chars[idx]

        phi_idx = min(int(phi * 10), len(self.phi_colors) - 1)
        color = self.phi_colors[phi_idx] if phi_idx >= 0 else ''
        reset = '\033[0m'

        self.buffer[y][x] = f"{color}{char}{reset}"

    def render_edge(self, x1: int, y1: int, x2: int, y2: int, weight: float):
        """
        Desenha uma aresta entre dois nós usando algoritmo de Bresenham.
        """
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            if 0 <= x1 < self.width and 0 <= y1 < self.height:
                edge_char = self.coherence_chars[int(weight * (len(self.coherence_chars)-1))]
                if self.buffer[y1][x1] == ' ':
                    self.buffer[y1][x1] = edge_char
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def clear(self):
        """Limpa o buffer."""
        for y in range(self.height):
            for x in range(self.width):
                self.buffer[y][x] = ' '

    def render(self, nodes: List[Tuple[int, int, float, float]], edges: List[Tuple[int, int, int, int, float]]):
        """Renderiza um frame completo."""
        self.clear()
        for x, y, coh, phi in nodes:
            self.render_node(x, y, coh, phi)
        for x1, y1, x2, y2, w in edges:
            self.render_edge(x1, y1, x2, y2, w)

        # Omitimos o clear screen real no ambiente CI para não poluir
        # os.system('clear')
        print("-" * self.width)
        for row in self.buffer:
            print(''.join(row))
        self.frame += 1

    def render_metrics(self, global_coherence: float, global_phi: float, entropy: float):
        """Exibe métricas globais abaixo do frame."""
        print(f"\nFrame: {self.frame} | C_global: {global_coherence:.3f} | Φ: {global_phi:.6f} | Entropia: {entropy:.3f} | Hz: 40")

def simulate_arkhe_ascii(iterations=10):
    renderer = ASCIIHypergraphRenderer(width=80, height=24)
    nodes = [
        (10, 12, 0.91, 0.0012),  # Alpha
        (30, 10, 0.89, 0.0010),  # Beta
        (50, 14, 0.88, 0.0009),  # Gamma
        (20, 20, 0.85, 0.0007),  # Delta
        (40, 18, 0.86, 0.0008),  # Epsilon
        (60, 16, 0.84, 0.0006),  # Zeta
        (70, 22, 0.94, 0.0063),  # Self
    ]
    edges = [
        (10, 12, 30, 10, 0.8), (30, 10, 50, 14, 0.7),
        (10, 12, 20, 20, 0.5), (60, 16, 70, 22, 0.9),
    ]

    for _ in range(iterations):
        renderer.render(nodes, edges)
        renderer.render_metrics(0.943, 0.006344, 0.158)
        time.sleep(0.025)

if __name__ == "__main__":
    simulate_arkhe_ascii()
